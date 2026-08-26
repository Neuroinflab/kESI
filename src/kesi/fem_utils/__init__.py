class MFEMNotInstalledError(ImportError):
    pass

def mfem_check():
    try:
        import mfem
    except ImportError:
        raise MFEMNotInstalledError("kESI is installed without MFEM support. "
                                    "To use this functionality install kESI "
                                    "with optional group fem: pip install kesi[fem]")
