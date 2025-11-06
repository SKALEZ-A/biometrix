from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from typing import Union
import logging

logger = logging.getLogger(__name__)

class BiometricEncryptor:
    def __init__(self, password: str = None, key_file: str = "config/encryption.key"):
        """Initialize with password-derived key or load from file."""
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                self.key = base64.urlsafe_b64decode(f.read())
        else:
            if not password:
                password = os.urandom(32).hex()  # Generate if none
            self._derive_key(password)
            with open(key_file, "wb") as f:
                f.write(base64.urlsafe_b64encode(self.key))
            logger.info("Generated new encryption key")
        
        self.cipher = Fernet(self.key)
    
    def _derive_key(self, password: str):
        """Derive key from password using PBKDF2."""
        salt = b'salt_for_biometrics'  # In prod, use random salt per key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        self.key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """Encrypt biometric data (embedding or hash)."""
        if isinstance(data, str):
            data = data.encode()
        encrypted = self.cipher.encrypt(data)
        logger.debug("Data encrypted successfully")
        return encrypted
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt stored biometric data."""
        try:
            decrypted = self.cipher.decrypt(encrypted_data)
            return decrypted
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError("Invalid encrypted data")
    
    def encrypt_embedding(self, embedding: list[float]) -> str:
        """Encrypt face/fingerprint embedding as JSON."""
        import json
        data = json.dumps(embedding).encode()
        return base64.b64encode(self.encrypt(data)).decode()
    
    def decrypt_embedding(self, encrypted_str: str) -> list[float]:
        """Decrypt and parse embedding."""
        import json
        encrypted_data = base64.b64decode(encrypted_str)
        decrypted = self.decrypt(encrypted_data)
        return json.loads(decrypted.decode())

# Key rotation stub
def rotate_key(old_password: str, new_password: str, encryptor: BiometricEncryptor):
    """Rotate encryption key (re-encrypt all data in prod)."""
    logger.info("Key rotation initiated")
    new_encryptor = BiometricEncryptor(new_password)
    # In full impl: Query DB, decrypt with old, encrypt with new
    logger.info("Key rotation complete (stub)")

# Global encryptor instance (load once)
encryptor = BiometricEncryptor()
