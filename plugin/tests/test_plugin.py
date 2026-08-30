from anemoi.inference.inputs import create_input
from anemoi.inference.testing import TestingContext


class _Checkpoint:
    # EkdInput.__init__ asks the checkpoint for a default namer.
    def default_namer(self):
        return lambda field, metadata: metadata.get("name")


class _Context(TestingContext):
    # Input.__init__ reads context.checkpoint; a stub with a namer is
    # enough to test registration and construction.
    checkpoint = _Checkpoint()


def test_plugin_registration(tmp_path):
    input_ = create_input(_Context(), {"raw": {"path": str(tmp_path), "variables": ["2t"]}})
    assert type(input_).__name__ == "RawInputPlugin"
    assert input_.path == tmp_path


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        from pathlib import Path

        test_plugin_registration(Path(tmp))
    print("ok")
