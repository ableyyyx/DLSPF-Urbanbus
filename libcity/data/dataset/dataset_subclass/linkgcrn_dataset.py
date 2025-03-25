import os
import pickle
from libcity.data.dataset.traffic_state_point_dataset import TrafficStatePointDataset


class LinkGCRNDataset(TrafficStatePointDataset):
    """
    自定义数据集类，用于 LinkGCRN 模型训练。
    除了加载 geo、dyna、rel 文件外，还加载一个 pkl 文件，
    其中存储了每条线路对应的站点索引列表（route_station_mapping）。
    该映射信息将保存在 data_feature 中，键名为 "route_station_mapping"，
    同时计算并保存站点数量（num_stations），以便在联合训练中正确构造预训练 STSANet 模型。

    配置文件中可以指定：
        route_mapping_file: 映射文件名称（默认 "route_station_mapping.pkl"）
    并要求配置参数 data_path 指向数据所在目录："/raw_data/URBAN_BUS_ROUTE"
    """

    def __init__(self, config):
        # 调用父类构造函数加载 geo、dyna、rel 等文件
        super().__init__(config)
        self.route_mapping_file = config.get('route_mapping_file', 'route_station_mapping')
        self.route_station_mapping = self._load_route_mapping(config)
        # 如果配置中没有直接提供站点数量，则从映射中自动推导
        self.num_stations = config.get('num_stations', None)
        if self.num_stations is None:
            max_station = 0
            for route in self.route_station_mapping:
                if route:
                    max_station = max(max_station, max(route))
            self.num_stations = max_station + 1
            self._logger.info(f"Calculated num_stations: {self.num_stations}")

    def _load_route_mapping(self, config):
        """加载 route_station_mapping.pkl 文件"""
        mapping_path = os.path.join(self.data_path, f'{self.route_mapping_file}.pkl')
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Route station mapping file not found at: {mapping_path}")
        with open(mapping_path, 'rb') as f:
            mapping = pickle.load(f)
        # 验证数据格式
        assert isinstance(mapping, list), "Mapping should be a list"
        assert len(mapping) == 308, "Expected 308 routes"
        self._logger.info(f"Loaded {len(mapping)} routes")
        return mapping

    def get_data_feature(self):
        # 先获取父类生成的特征字典
        feature = super().get_data_feature()
        # 添加 route_station_mapping 信息和站点数量
        feature['route_station_mapping'] = self.route_station_mapping
        feature['num_stations'] = self.num_stations
        return feature
