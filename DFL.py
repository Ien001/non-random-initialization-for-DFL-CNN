import torch
import torch.nn as  nn
import torch.nn.functional as F
import torchvision

class DFL_VGG16(nn.Module):
	def __init__(self, k = 10, nclass = 200):
		super(DFL_VGG16, self).__init__()
		self.k = k
		self.nclass = nclass
		
		# k channels for one class, nclass is total classes, therefore k * nclass for conv6
		vgg16featuremap = torchvision.models.vgg16_bn(pretrained=True).features
		conv1_conv4 = torch.nn.Sequential(*list(vgg16featuremap.children())[:-11])
		conv5 = torch.nn.Sequential(*list(vgg16featuremap.children())[-11:])
		conv6 = torch.nn.Conv2d(512, k * nclass, kernel_size = 1, stride = 1, padding = 0)
		pool6 = torch.nn.MaxPool2d((56, 56), stride = (56, 56), return_indices = True)

		# Feature extraction root
		self.conv1_conv4 = conv1_conv4

		# G-Stream
		self.conv5 = conv5
		self.cls5 = nn.Sequential(
			nn.Conv2d(512, 200, kernel_size=1, stride = 1, padding = 0),
			nn.BatchNorm2d(200),
			nn.ReLU(True),
			nn.AdaptiveAvgPool2d((1,1)),
			)

		# P-Stream
		self.conv6 = conv6
		self.pool6 = pool6
		self.cls6 = nn.Sequential(
			nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
			nn.AdaptiveAvgPool2d((1,1)),
			)

		# Side-branch
		self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)

	def forward(self, x):
		batchsize = x.size(0)

		# Stem: Feature extractionc
		inter4 = self.conv1_conv4(x)
		#print(inter4.shape)

        # G-stream
		x_g = self.conv5(inter4)
		out1 = self.cls5(x_g)
		out1 = out1.view(batchsize, -1)

		# P-stream ,indices is for visualization
		x_p = self.conv6(inter4)
		x_p, indices = self.pool6(x_p)
		inter6 = x_p
		out2 = self.cls6(x_p)
		out2 = out2.view(batchsize, -1)
		
		# Side-branch
		inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
		out3 = self.cross_channel_pool(inter6)
		out3 = out3.view(batchsize, -1)
	
		return out1, out2, out3, indices


class DFL_ResNet_for_sample(nn.Module):
	def __init__(self, k = 10, nclass = 200):
		super(DFL_ResNet_for_sample, self).__init__()
		self.k = k
		self.nclass = nclass

		# k channels for one class, nclass is total classes, therefore k * nclass for conv6
		resnet50 = torchvision.models.resnet50(pretrained=True)
		# conv1_conv4
		layers_conv1_conv4 = [
		resnet50.conv1,
		resnet50.bn1,
		resnet50.relu,
		resnet50.maxpool,
		]
		for i in range(3):
			name = 'layer%d' % (i + 1)
			layers_conv1_conv4.append(getattr(resnet50, name))
		conv1_conv4 = torch.nn.Sequential(*layers_conv1_conv4)

		self.conv1_conv4 = conv1_conv4

	def forward(self, x):
		batchsize = x.size(0)

		# Stem: Feature extraction
		inter4 = self.conv1_conv4(x)
		center = torch.norm(inter4.norm(2,0),2,0).mean()

		return center


class DFL_ResNet(nn.Module):
	def __init__(self, k = 10, nclass = 200):
		super(DFL_ResNet, self).__init__()
		self.k = k
		self.nclass = nclass

		# k channels for one class, nclass is total classes, therefore k * nclass for conv6
		resnet50 = torchvision.models.resnet50(pretrained=True)
		# conv1_conv4
		layers_conv1_conv4 = [
		resnet50.conv1,
		resnet50.bn1,
		resnet50.relu,
		resnet50.maxpool,
		]
		for i in range(3):
			name = 'layer%d' % (i + 1)
			layers_conv1_conv4.append(getattr(resnet50, name))
		conv1_conv4 = torch.nn.Sequential(*layers_conv1_conv4)

		# conv5
		layers_conv5 = []
		layers_conv5.append(getattr(resnet50, 'layer4'))
		conv5 = torch.nn.Sequential(*layers_conv5)

		conv6 = torch.nn.Conv2d(1024, k * nclass, kernel_size = 1, stride = 1, padding = 0)
		pool6 = torch.nn.MaxPool2d((28, 28), stride = (28, 28), return_indices = True)

		# Feature extraction root
		self.conv1_conv4 = conv1_conv4

		# G-Stream
		self.conv5 = conv5
		self.cls5 = nn.Sequential(
			nn.Conv2d(2048, 200, kernel_size=1, stride = 1, padding = 0),
			nn.BatchNorm2d(200),
			nn.ReLU(True),
			nn.AdaptiveAvgPool2d((1,1)),
			)

		# P-Stream
		self.conv6 = conv6
		self.pool6 = pool6
		self.cls6 = nn.Sequential(
			nn.Conv2d(k * nclass, nclass, kernel_size = 1, stride = 1, padding = 0),
			nn.AdaptiveAvgPool2d((1,1)),
			)

		# Side-branch
		self.cross_channel_pool = nn.AvgPool1d(kernel_size = k, stride = k, padding = 0)

	def forward(self, x):
		batchsize = x.size(0)

		# Stem: Feature extraction
		inter4 = self.conv1_conv4(x)
		#print('inter4',inter4.shape)

        # G-stream
		#print('inter4',inter4.shape)
		x_g = self.conv5(inter4)
		out1 = self.cls5(x_g)
		out1 = out1.view(batchsize, -1)
		#print('out1',out1.shape)

		# P-stream ,indices is for visualization
		x_p = self.conv6(inter4)
		#print('conv6',x_p.shape)
		x_p, indices = self.pool6(x_p)
		#print(x_p.shape)
		inter6 = x_p
		out2 = self.cls6(x_p)
		out2 = out2.view(batchsize, -1)
		#print('out2',out2.shape)
		
		# Side-branch
		inter6 = inter6.view(batchsize, -1, self.k * self.nclass)
		out3 = self.cross_channel_pool(inter6)
		out3 = out3.view(batchsize, -1)
		#print('out3',out3.shape)

	
		return out1, out2, out3, indices

if __name__ == '__main__':
	input_test = torch.ones(1,10,3,448,448)
	net = DFL_ResNet()
	output_test = net(input_test)
	print(output_test)



