import inspect
import pytest
from openai import OpenAI

from generate_synthetic_ocr import get_chat_request


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def test_gpt_default_model_still_exists():
    """Test that the default model is in the list of available OpenAI models, and has not been removed by OpenAI."""
    client = OpenAI()
    default_model = get_default_args(get_chat_request)["model"]
    current_openai_models_lst = [model.id for model in client.models.list()]
    assert (
        default_model in current_openai_models_lst
    ), f"Default model should be a string in the list of available openai models: {current_openai_models_lst}"
