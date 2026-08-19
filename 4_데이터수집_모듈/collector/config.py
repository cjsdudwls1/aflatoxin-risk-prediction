# -*- coding: utf-8 -*-
"""config.yaml 로더 및 서비스키 관리."""
import os

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError("pyyaml 필요: pip install pyyaml") from e

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                       # 모듈 루트(아플라톡신_데이터수집_모듈)
DEFAULT_CONFIG_PATH = os.path.join(_ROOT, 'config.yaml')


class Config:
    """config.yaml 내용을 감싼 접근자."""

    def __init__(self, data, path):
        self._d = data or {}
        self.path = path

    @property
    def api_keys(self):
        return self._d.get('api_keys', {}) or {}

    def key(self, name):
        """필수 키. 없으면 명확한 에러."""
        v = self.api_keys.get(name, '')
        if not v:
            raise KeyError(
                f"config.yaml 의 api_keys.{name} 가 비어있습니다. "
                f"({self.path}) data.go.kr 에서 키를 발급해 채우세요."
            )
        return v

    def optional_key(self, name):
        """선택 키. 없으면 None."""
        return self.api_keys.get(name) or None

    @property
    def defaults(self):
        return self._d.get('defaults', {}) or {}

    def default(self, name, fallback=None):
        return self.defaults.get(name, fallback)


def load_config(path=None):
    """config.yaml 로드. 경로 미지정시 모듈 루트의 config.yaml."""
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"설정파일이 없습니다: {path}\n"
            f"  -> config.example.yaml 을 config.yaml 로 복사한 뒤 키를 채우세요."
        )
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return Config(data, path)
