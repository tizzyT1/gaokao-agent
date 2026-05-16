from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_pro_model: str = "deepseek-v4-pro"
    deepseek_flash_model: str = "deepseek-v4-flash"
    backend_api_url: str = "http://127.0.0.1:8000"
    backend_timeout: float = 30.0


settings = Settings()
