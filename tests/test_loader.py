import subprocess
import sys
from textwrap import dedent


def test_load_plugins_registers_mistral():
    code ="""

    from app.core.loader import load_plugins
    from app.core.factory import ProviderFactory

    load_plugins()
    provider = ProviderFactory.build("mistral")

    assert type(provider).__name__ == "MistralProvider"

    """
    subprocess.run([sys.executable, "-c", dedent(code)], check=True)
