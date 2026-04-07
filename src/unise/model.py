import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks
from pathlib import Path
import soundfile as sf
import time
import math
import random

from .bicodec import BiCodecTokenizer
from .llm import LLM_SFT


from transformers import AutoModel


# Lazy import for pytorch_lightning — make it optional
try:
    import pytorch_lightning as pl
    _HAS_PL = True
except ImportError:
    _HAS_PL = False


# If pytorch_lightning is available, Model inherits from pl.LightningModule
# so it can be used with Trainer. Otherwise it falls back to nn.Module.
if _HAS_PL:
    _BaseClass = pl.LightningModule
else:
    _BaseClass = nn.Module


class Model(_BaseClass):
    def __init__(self, config):
        super().__init__()
        if _HAS_PL:
            self.save_hyperparameters()
        self.config = config
        self.stft_conf = config['stft_config']
        
        self.tokenizer = BiCodecTokenizer(model_dir=config['codec_ckpt_dir'])
        self.dnn = LLM_SFT(**config['llm_config'])

        self.semantic_model = AutoModel.from_pretrained("microsoft/wavlm-base-plus").eval()
        self.semantic_model.requires_grad_(False)

        self.current_traning_step = -1

    @torch.no_grad()
    def extract_semantic_features(self, wavs: torch.Tensor) -> torch.Tensor:
        """extract wav2vec2 features"""
        # wavs: (b,t)
        wavs = F.pad(wavs, (160, 160))
        
        feats = self.semantic_model(wavs, output_hidden_states=True)
        feats_mix = torch.stack(feats.hidden_states, dim=1).mean(1)

        return feats_mix.detach()
    
    def stft_logmel(self, x):
        # x:(B,T)
        assert x.ndim == 2
        hop_length = self.stft_conf['hop_length']
        win_length = self.stft_conf['win_length']
        n_fft = self.stft_conf['n_fft']
        n_mels = self.stft_conf['n_mels']

        pad_length = math.ceil(x.size(-1) / hop_length) * hop_length - x.size(-1)
        x = torch.nn.functional.pad(x, ((win_length - hop_length) // 2, pad_length + (win_length - hop_length) // 2))
        spec = torch.stft(
            x,
            n_fft,
            hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(x.device),
            onesided=True,
            center=False,
            return_complex=True,
        ).transpose(1, 2)  # (B,T,F)
        if not hasattr(self, 'fb'):
            fb = melscale_fbanks(n_freqs=n_fft // 2 + 1, f_min=0.0, f_max=8000.0, n_mels=n_mels, sample_rate=16000)
            setattr(self, 'fb', fb.to(x.device))
        mag = spec.abs()  # (b,t,f)
        mel = mag @ self.fb  # (B,T,M)
        mel = torch.log(mel + 1e-10)
        return mel
    
    # state_dict: exclude tokenizer and semantic_model
    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        for key in list(state.keys()):
            if key.startswith('tokenizer.') or key.startswith('semantic_model.'):
                del state[key]
        return state

    # load_state_dict: exclude tokenizer
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=False)
    
    def forward(self, batch):
        pass

    # ------------------------------------------------------------------
    # PyTorch Lightning methods (only used when PL is available)
    # ------------------------------------------------------------------
    if _HAS_PL:
        def training_step(self, batch, batch_idx):
            mode, enroll, mix, speech, interf, fs, lengths, names = batch

            if mode == 'rtse':
                global_tokens, semantic_tokens = self.tokenizer.tokenize(interf)
            else:
                global_tokens, semantic_tokens = self.tokenizer.tokenize(speech)
            
            mix_mel = self.stft_logmel(mix)
            mix_feats = self.extract_semantic_features(mix)
            if enroll is not None:
                enroll_mel = self.stft_logmel(enroll)
                enroll_feats = self.extract_semantic_features(enroll)
            else:
                enroll_mel, enroll_feats = None, None

            loss, acc = self.dnn(
                task_name = mode,
                enroll_mel = enroll_mel,
                enroll_feats = enroll_feats,
                mix_mel = mix_mel,
                mix_feats = mix_feats,
                global_ids = global_tokens.squeeze(1), 
                semantic_ids = semantic_tokens,
            )
            
            self.log_dict({'train_loss': loss, 'train_acc': acc}, on_step=True, on_epoch=True, prog_bar=True)
            self.current_traning_step += 1
            return loss

        def on_train_epoch_end(self):
            pass

        def validation_step(self, batch, batch_idx):
            mode, enroll, mix, speech, interf, fs, lengths, names = batch

            if mode == 'rtse':
                global_tokens, semantic_tokens = self.tokenizer.tokenize(interf)
            else:
                global_tokens, semantic_tokens = self.tokenizer.tokenize(speech)
            
            mix_mel = self.stft_logmel(mix)
            mix_feats = self.extract_semantic_features(mix)
            if enroll is not None:
                enroll_mel = self.stft_logmel(enroll)
                enroll_feats = self.extract_semantic_features(enroll)
            else:
                enroll_mel, enroll_feats = None, None

            loss, acc = self.dnn(
                task_name = mode,
                enroll_mel = enroll_mel,
                enroll_feats = enroll_feats,
                mix_mel = mix_mel,
                mix_feats = mix_feats,
                global_ids = global_tokens.squeeze(1), 
                semantic_ids = semantic_tokens,
            )
            
            self.log_dict({'valid_loss': loss, 'valid_acc': acc}, on_step=False, on_epoch=True, sync_dist=True)
        
        def on_validation_epoch_end(self):
            if not self.trainer.sanity_checking:
                epoch = self.current_epoch
                step = self.current_traning_step
                ckpt_name = f'epoch={epoch}-step={step}.ckpt'
                self.trainer.save_checkpoint(self.config['ckpt_dir'] / ckpt_name)
        
        def test_step(self, batch, batch_idx):
            mode, enroll, src, tgt, fs, lengths, names = batch

            do_sample = False
            if mode == 'se':
                seg_len = 5 * 16000
                pad_len = math.ceil(src.size(-1) / seg_len) * seg_len - src.size(-1)
                seg_src = np.pad(src.cpu().numpy(), [(0, 0), (0, pad_len)], 'wrap')
                seg_src = torch.from_numpy(seg_src).to(src.device)
                seg_src = seg_src.reshape(-1, seg_len)
                seg_src = seg_src / src.abs().max(dim=-1, keepdim=True)[0]

                mix_mel = self.stft_logmel(seg_src)
                mix_feats = self.extract_semantic_features(seg_src)
                global_ids, semantic_ids = self.dnn.generate(
                    task_name='se',
                    enroll_mel=None,
                    enroll_feats=None,
                    mix_mel=mix_mel,
                    mix_feats=mix_feats,
                    do_sample=do_sample,
                )
                est = self.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids).squeeze(1)
                est = est.reshape(-1)[:src.size(-1)]
                est = est.cpu().numpy()

                if 'save_enhanced' in self.config and self.config['save_enhanced'] is not None:
                    sf.write(Path(self.config['save_enhanced']) / f'{names[0]}.wav', est, samplerate=int(fs[0]))
            elif mode == 'tse':
                seg_len = 5 * 16000
                pad_len = math.ceil(src.size(-1) / seg_len) * seg_len - src.size(-1)
                seg_src = np.pad(src.cpu().numpy(), [(0, 0), (0, pad_len)], 'wrap')
                seg_src = torch.from_numpy(seg_src).to(src.device)
                seg_src = seg_src.reshape(-1, seg_len)

                enroll_mel = self.stft_logmel(enroll)
                enroll_feats = self.extract_semantic_features(enroll)
                enroll_mel = torch.cat([enroll_mel for _ in range(seg_src.size(0))], dim=0)
                enroll_feats = torch.cat([enroll_feats for _ in range(seg_src.size(0))], dim=0)

                mix_mel = self.stft_logmel(seg_src)
                mix_feats = self.extract_semantic_features(seg_src)
                global_ids, semantic_ids = self.dnn.generate(
                    task_name='tse',
                    enroll_mel=enroll_mel,
                    enroll_feats=enroll_feats,
                    mix_mel=mix_mel,
                    mix_feats=mix_feats,
                    do_sample=do_sample,
                )

                est = self.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids).squeeze(1)
                est = est.reshape(-1)[:src.size(-1)]
                est = est.cpu().numpy()

                if 'save_enhanced' in self.config and self.config['save_enhanced'] is not None:
                    sf.write(Path(self.config['save_enhanced']) / f'{names[0]}.wav', est, samplerate=int(fs[0]))
            elif mode == 'ss':
                seg_len = 5 * 16000
                if src.size(-1) > seg_len:
                    seg_src = src[:, :seg_len]
                else:
                    seg_src = np.pad(src.cpu().numpy(), [(0, 0), (0, seg_len - src.size(-1))], 'wrap')
                    seg_src = torch.from_numpy(seg_src).to(src.device)
                
                mix_mel = self.stft_logmel(seg_src)
                mix_feats = self.extract_semantic_features(seg_src)
                global_ids, semantic_ids = self.dnn.generate(
                    task_name='se',
                    enroll_mel=None,
                    enroll_feats=None,
                    mix_mel=mix_mel,
                    mix_feats=mix_feats,
                    do_sample=do_sample,
                )
                enroll = self.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids).squeeze(1)
                enroll = enroll[:, :seg_len]
                enroll = enroll / (torch.max(torch.abs(enroll)) + 1e-5) * 0.99
                enroll_mel = self.stft_logmel(enroll)
                enroll_feats = self.extract_semantic_features(enroll)
                

                pad_len = math.ceil(src.size(-1) / seg_len) * seg_len - src.size(-1)
                seg_src = np.pad(src.cpu().numpy(), [(0, 0), (0, pad_len)], 'wrap')
                seg_src = torch.from_numpy(seg_src).to(src.device)
                seg_src = seg_src.reshape(-1, seg_len)
                enroll_mel = torch.cat([enroll_mel for _ in range(seg_src.size(0))], dim=0)
                enroll_feats = torch.cat([enroll_feats for _ in range(seg_src.size(0))], dim=0)
                mix_mel = self.stft_logmel(seg_src)
                mix_feats = self.extract_semantic_features(seg_src)
                global_ids, semantic_ids = self.dnn.generate(
                    task_name='tse',
                    enroll_mel=enroll_mel,
                    enroll_feats=enroll_feats,
                    mix_mel=mix_mel,
                    mix_feats=mix_feats,
                    do_sample=do_sample,
                )
                est = self.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids).squeeze(1)
                est = est.reshape(-1)[:src.size(-1)].cpu().numpy()
                if 'save_enhanced' in self.config and self.config['save_enhanced'] is not None:
                    sf.write(Path(self.config['save_enhanced']) / f'{names[0]}_s1.wav', est, samplerate=int(fs[0]))
                

                global_ids, semantic_ids = self.dnn.generate(
                    task_name='rtse',
                    enroll_mel=enroll_mel,
                    enroll_feats=enroll_feats,
                    mix_mel=mix_mel,
                    mix_feats=mix_feats,
                    do_sample=do_sample,
                )
                est = self.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids).squeeze(1)
                est = est.reshape(-1)[:src.size(-1)].cpu().numpy()
                if 'save_enhanced' in self.config and self.config['save_enhanced'] is not None:
                    sf.write(Path(self.config['save_enhanced']) / f'{names[0]}_s2.wav', est, samplerate=int(fs[0]))


        def on_test_epoch_end(self):
            pass

        @torch.inference_mode()
        def generate(self, task_name, enroll, mixture):
            # cond: (B, T)
            if enroll is not None:
                enroll = self.stft_logmel(enroll)
            length = mixture.size(-1)
            mixture = self.stft_logmel(mixture)

            global_ids, semantic_ids = self.dnn.generate(
                task_name=task_name,
                enroll=enroll,
                mixture=mixture,
                do_sample=True,
            )
            wav_rec = self.tokenizer.detokenize(global_ids.unsqueeze(1), semantic_ids)[...,:length]

            import soundfile as sf
            sf.write('test.wav', wav_rec.squeeze().cpu().numpy(), 16000)


        def on_save_checkpoint(self, ckpt):
            ckpt['current_traning_step'] = self.current_traning_step
        
        def on_load_checkpoint(self, ckpt):
            self.current_traning_step = ckpt['current_traning_step']

        def configure_optimizers(self):
            opt = torch.optim.AdamW(self.dnn.parameters(), **self.config['opt'])

            def warmup_lambda(step):
                warmup_steps = self.config['sch']['warmup_steps']
                step_decay = self.config['sch']['step_decay']
                if step < warmup_steps:
                    return 0.5 * (1 + math.cos(math.pi * (1 - step / warmup_steps)))
                else:
                    return max(step_decay ** (step - warmup_steps), self.config['sch']['min_factor'])
                
            sch = {
                'scheduler': torch.optim.lr_scheduler.LambdaLR(opt, warmup_lambda), 
                'interval': 'step',
                'frequency': 1,
            }

            return [opt], [sch]
