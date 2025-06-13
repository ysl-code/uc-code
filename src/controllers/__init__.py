REGISTRY = {}

from .basic_controller import BasicMAC
from .n_controller import NMAC
from .ppo_controller import PPOMAC
from .conv_controller import ConvMAC
from .basic_central_controller import CentralBasicMAC
from .lica_controller import LICAMAC
from .dop_controller import DOPMAC
from .mcga_n_controller import MCGA_NMAC
from .cola_n_controller import COLANMAC
from .ducc_n_controller import DUCCNMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["n_mac"] = NMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["dop_mac"] = DOPMAC
REGISTRY["mcga_n_mac"] = MCGA_NMAC
REGISTRY["cola_n_mac"] = COLANMAC
REGISTRY["ducc_n_mac"] = DUCCNMAC

from .uc_n_controller import UCNMAC
REGISTRY["uc_n_mac"] = UCNMAC

