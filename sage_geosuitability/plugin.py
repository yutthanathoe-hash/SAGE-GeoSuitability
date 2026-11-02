from qgis.core import QgsApplication
from .provider import SAGEGeoSuitabilityProvider

class SAGEGeoSuitabilityPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.provider = None

    def initGui(self):
        if self.provider is None:
            self.provider = SAGEGeoSuitabilityProvider()
            QgsApplication.processingRegistry().addProvider(self.provider)

    def unload(self):
        if self.provider is not None:
            QgsApplication.processingRegistry().removeProvider(self.provider)
            self.provider = None
