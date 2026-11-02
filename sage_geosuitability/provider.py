from qgis.core import QgsProcessingProvider
from .algorithm import SAGEGeoSuitability

class SAGEGeoSuitabilityProvider(QgsProcessingProvider):
    def id(self):
        return 'sage_geosuitability'

    def name(self):
        return 'SAGE-GeoSuitability'

    def longName(self):
        return 'SAGE-GeoSuitability — Universal Scientific Spatial ML'

    def loadAlgorithms(self):
        self.addAlgorithm(SAGEGeoSuitability())
