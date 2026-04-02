"""
Internationalization (i18n) module for Gradio UI
Supports multiple languages with easy translation management
"""
import os
import json
from typing import Dict, Optional


class I18n:
    """Internationalization handler"""
    
    def __init__(self, default_language: str = "en"):
        """
        Initialize i18n handler
        
        Args:
            default_language: Default language code (en, zh, ja, etc.)
        """
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        self._load_all_translations()
    
    def _load_all_translations(self):
        """Load all translation files from i18n directory"""
        current_file = os.path.abspath(__file__)
        module_dir = os.path.dirname(current_file)
        i18n_dir = os.path.join(module_dir, "i18n")
        
        if not os.path.exists(i18n_dir):
            # Create i18n directory if it doesn't exist
            os.makedirs(i18n_dir)
            return
        
        # Load all JSON files in i18n directory
        for filename in os.listdir(i18n_dir):
            if filename.endswith(".json"):
                lang_code = filename[:-5]  # Remove .json extension
                filepath = os.path.join(i18n_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.translations[lang_code] = json.load(f)
                except Exception as e:
                    print(f"Error loading translation file {filename}: {e}")
    
    def set_language(self, language: str):
        """Set current language"""
        if language in self.translations:
            self.current_language = language
        else:
            print(f"Warning: Language '{language}' not found, using default")
    
    def t(self, key: str, **kwargs) -> str:
        """
        Translate a key to current language
        
        Args:
            key: Translation key (dot-separated for nested keys)
            **kwargs: Optional format parameters
        
        Returns:
            Translated string
        """
        # Get translation from current language
        translation = self._get_nested_value(
            self.translations.get(self.current_language, {}), 
            key
        )
        
        # Fallback to English if not found
        if translation is None:
            translation = self._get_nested_value(
                self.translations.get('en', {}), 
                key
            )
        
        # Final fallback to key itself
        if translation is None:
            translation = key
        
        # Apply formatting if kwargs provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except KeyError:
                pass
        
        return translation
    
    def _get_nested_value(self, data: dict, key: str) -> Optional[str]:
        """
        Get nested dictionary value using dot notation
        
        Args:
            data: Dictionary to search
            key: Dot-separated key (e.g., "section.subsection.key")
        
        Returns:
            Value if found, None otherwise
        """
        keys = key.split('.')
        current = data
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current if isinstance(current, str) else None
    
    def get_available_languages(self) -> list:
        """Get list of available language codes"""
        return list(self.translations.keys())


# Global i18n instance
_i18n_instance: Optional[I18n] = None


def get_i18n(language: Optional[str] = None) -> I18n:
    """
    Get global i18n instance
    
    Args:
        language: Optional language to set
    
    Returns:
        I18n instance
    """
    global _i18n_instance
    
    if _i18n_instance is None:
        _i18n_instance = I18n(default_language=language or "en")
    elif language is not None:
        _i18n_instance.set_language(language)
    
    return _i18n_instance


def t(key: str, **kwargs) -> str:
    """
    Convenience function for translation
    
    Args:
        key: Translation key
        **kwargs: Optional format parameters
    
    Returns:
        Translated string
    """
    return get_i18n().t(key, **kwargs)
