from re import M
from .ST_GCN_Block import ST_GCN_18


# import logging

# from .graph import Graph
# from .ntu_feeder import NTU_Feeder, NTU_Location_Feeder
# from .sbu_feeder import SBU_Feeder

# __data_args = {
#     'AIDE': {'class': 4, 'shape': [16, 3, 16, 21, 1]},
# }

# def create(dataset, **kwargs):
#     g = Graph(dataset, **kwargs)
#     try:
#         data_args = __data_args[dataset]
#         num_class = data_args['class']
#     except:
#         logging.info('')
#         logging.error('Error: Do NOT exist this dataset: {}!'.format(dataset))
#         raise ValueError()

#     # feeders = {
#     #     'train': data_args['feeder'](dataset=dataset, phase='train', connect_joint=g.connect_joint, **kwargs),
#     #     'eval' : data_args['feeder'](dataset=dataset, phase='eval', connect_joint=g.connect_joint, **kwargs),
#     # }
#     data_shape = __data_args['shape']
#     in_channels = data_shape[1]
#     # if 'ntu' in dataset:
#     #     feeders.update({'location': NTU_Location_Feeder(data_shape)})
#     return in_channels, data_shape, num_class, g.A

# def set_datashape(self):
#     data_shape = [16, 3, 16, 21, 1]
#     data_shape[0] = len(self.inputs) if self.inputs.isupper() else 1
#     data_shape[1] = 74 if self.inputs in ['joint','joint_motion','bone','bone_motion'] else 36
#     data_shape[2] = 64 if self.crop else self.T
#     if 'mutual' in self.graph:
#         data_shape[3] = data_shape[3]*data_shape[4]
#         data_shape[4] = 1
#     if self.processing in ['symmetry','padding']: 
#         assert data_shape[4] == 1
#         data_shape[4] = data_shape[4]*2
#     return data_shape

# self.model = model.create(self.args.model_type, **(self.args.model_args), **kwargs)
# data_shape, num_class, g.A=

# kwargs = {
#             'data_shape': self.data_shape,
#             'num_class': self.num_class,
#             'A': torch.Tensor(self.A),
#             'parts': self.parts,
#         }
# def channel_update(dataset):

#     kwargs.update({
#         'in_channels': dataset.create()[0][1],
#     })
#     return ST_GCN_18(**kwargs)