class MMH3Error(Exception):
    """Base error for MMH3_MEDIA."""


class MMH3FormatError(MMH3Error):
    """Archive/manifest is invalid or unsafe."""


class MMH3ResourceError(MMH3Error):
    """Resource selection, type, or payload is invalid."""


class MMH3IntegrityError(MMH3Error):
    """A checksum or size verification failed."""
