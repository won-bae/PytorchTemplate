
model=dict(
  num_classes=10,
  backbone='resnet18',
  pretrained='true',
  checkpoint_path=''
)

train=dict(
  num_epochs=100,

  optimizer=dict(
    name='sgd',
    lr=0.01,
    momentum=0.9,
    nesterov=False,
    weight_decay=0.001),

  lr_schedule=dict(
    name='custom',
    milestones=[50, 100],
    gamma=0.3)
)

eval=dict(
  standard='accuracy')

data=dict(
  name='cifar10',
  root='data',
  batch_size=128,
  num_workers=4,

  transform=dict(
    image_size=32,
    crop_size=28)
)
