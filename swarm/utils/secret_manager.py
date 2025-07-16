#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVANCED SECRET MANAGER
Version: 2.0.0
Created: 2025-07-17
Author: Security Team
"""

from cryptography.fernet import Fernet
import os
import base64
import logging
from swarm.utils import Logger

logger = Logger(name="SecretManager")

class SecretManager:
    def __init__(self, key=None):
        # Load encryption key from environment or generate new
        self.key = key or os.getenv("ENCRYPTION_KEY")
        
        if not self.key:
            logger.warning("No encryption key found. Generating temporary key.")
            self.key = Fernet.generate_key().decode()
            logger.info("For production, set ENCRYPTION_KEY environment variable")
        
        self.cipher = Fernet(self.key.encode())
    
    def encrypt(self, plaintext: str) -> str:
        """Encrypt sensitive data"""
        if not plaintext:
            return ""
            
        encrypted = self.cipher.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        """Decrypt sensitive data"""
        if not ciphertext:
            return ""
            
        try:
            decoded = base64.urlsafe_b64decode(ciphertext.encode())
            return self.cipher.decrypt(decoded).decode()
        except Exception as e:
            logger.error(f"Decryption failed: {str(e)}")
            raise ValueError("Invalid ciphertext or key")
    
    def secure_env(self, env_dict: dict) -> dict:
        """Encrypt all values in environment dictionary"""
        return {k: self.encrypt(v) for k, v in env_dict.items()}
    
    def load_secure_env(self, encrypted_dict: dict) -> dict:
        """Decrypt all values in environment dictionary"""
        return {k: self.decrypt(v) for k, v in encrypted_dict.items()}
    
    def rotate_key(self, new_key: str):
        """Rotate to a new encryption key"""
        logger.info("Initiating key rotation")
        self.key = new_key
        self.cipher = Fernet(self.key.encode())
        logger.info("Encryption key rotated successfully")
    
    def save_to_vault(self, secrets: dict, vault_path: str = "secrets.vault"):
        """Save encrypted secrets to file"""
        try:
            encrypted = self.secure_env(secrets)
            with open(vault_path, 'w') as f:
                json.dump(encrypted, f)
            logger.info(f"Secrets saved to {vault_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save secrets: {str(e)}")
            return False
    
    def load_from_vault(self, vault_path: str = "secrets.vault") -> dict:
        """Load encrypted secrets from file"""
        try:
            with open(vault_path, 'r') as f:
                encrypted = json.load(f)
            return self.load_secure_env(encrypted)
        except Exception as e:
            logger.error(f"Failed to load secrets: {str(e)}")
            return {}
