import os
from functools import lru_cache
from dotenv import load_dotenv

DEFAULT_REQUIRED_KEYS = ("OPENAI_API_KEY", "GOOGLE_API_KEY")

@lru_cache(maxsize=1)
def configure_environment(required_keys=None):
    """
    Factory function to configure environment variables.
    Executes once and caches results.
    """
    if required_keys is None:
        required_keys = ("OPENAI_API_KEY", "GOOGLE_API_KEY")

    
    print("Configuring for local environment...")
    load_dotenv()

    # Validation
    for key in required_keys:
        if not os.getenv(key):
            raise ValueError(f"Missing required environment variable: {key}")

    return True