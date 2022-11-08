#pylint: disable=unused-import

from pkg_resources import resource_string

__version__ = resource_string(__name__, "VERSION").decode("ascii")
__version_info__ = __version__.split(".")

from my_nas.utils import RegistryMeta
from my_nas.base import Component
from my_nas import evaluator
