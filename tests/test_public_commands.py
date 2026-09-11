import importlib


PUBLIC_MODULES = [
    "dockpost.init",
    "dockpost.minimize",
    "dockpost.md",
    "dockpost.qc",
    "dockpost.enrich",
    "dockpost.landscape",
    "dockpost.strain",
    "dockpost.master",
]


def test_public_modules_import():

    for module_name in PUBLIC_MODULES:

        module = importlib.import_module(
            module_name
        )

        assert hasattr(
            module,
            "main",
        )
