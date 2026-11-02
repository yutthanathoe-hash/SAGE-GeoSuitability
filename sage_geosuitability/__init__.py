from .plugin import SAGEGeoSuitabilityPlugin

def classFactory(iface):
    return SAGEGeoSuitabilityPlugin(iface)
