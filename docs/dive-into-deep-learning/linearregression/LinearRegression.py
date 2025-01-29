from manim import *
import numpy as np
import torch
import random

class LinearRegression(Scene):
    def construct(self):
        def data_generator(w,b,num):
            X = torch.normal(0, 1, (num, len(w)))
            y = torch.matmul(X, w) + b
            y += torch.normal(0, 0.01, y.shape)
            return X, y.reshape((-1, 1))

        true_w = torch.tensor([2,-3.4])
        true_b = 4.2
        features, labels = data_generator(true_w,true_b,1000)
        normal_data = features[:,[0]].numpy()
        #plt.hist(normal_data,bins=100,density=True,color='lightblue')  
         

        head = Text("Linear Regression - Buezwqwg")
        head.set_color(BLUE)
        self.play(Create(head))

        head_0 = Text("In one process of Linear Regression, there bascially includes 5 steps",font_size=30)
        self.play(Uncreate(head),Write(head_0))
        self.play(head_0.animate.move_to(UP*3.5))
        head_1 = Text("1. Initial Parameters",font_size=30)
        head_2 = Text("2. Defining Model and Loss Function",font_size=30)
        head_3 = Text("3. Optimization",font_size=30)
        head_4 = Text("4. Loop",font_size=30)
        head = VGroup(head_1,head_2,head_3,head_4)
        head.arrange(DOWN)
        self.play(Write(head))

        # --------------------------------------------------------------------------------------------

        head_5 = Text("In this animate, we start with generating the data",font_size=30)
        head_5.move_to(UP*3.5)
        self.play(Uncreate(head),Uncreate(head_0),Write(head_5))
        code_text = '''
        def data_generator(w, b, num):
            X = torch.normal(0, 1, (num, len(w)))
            y = torch.matmul(X, w) + b
            y += torch.normal(0, 0.01, y.shape)
            return X, y.reshape((-1, 1))
            
        true_w = torch.tensor([2,-3.4])
        true_b = 4.2
        features, labels = data_generator(true_w,true_b,1000)
        '''
        code = Code(code=code_text,insert_line_no=False,language="Python",font="Monospace")
        self.play(Write(code),Uncreate(head_5),run_time=3)
        self.wait(3)
        self.play(Unwrite(code))
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[0, 0.5, 0.1],
            axis_config={"color": BLUE},
        ).add_coordinates()

        # 正态分布函数 y = (1/sqrt(2*pi)) * exp(-x^2 / 2)
        normal_curve = axes.plot(
            lambda x: (1 / (2 * PI) ** 0.5) * np.exp(-x**2 / 2),
            color=YELLOW
        )

        # 绘制均值为0的竖线
        mean_line = DashedLine(
            start=axes.c2p(0, 0),
            end=axes.c2p(0, (1 / (2 * PI) ** 0.5)),
            color=RED
        )

        # 添加图形和标注
        self.play(Create(axes))
        self.play(Create(normal_curve), Create(mean_line))
        
        # 标注均值和标准差
        mean_label = MathTex(r"\mu=0").next_to(mean_line, DOWN)
        std_label = MathTex(r"\sigma=1").next_to(normal_curve, UP, buff=0.5)
        self.play(Write(mean_label), Write(std_label))

        # 展示最终效果
        self.wait(2)
        self.play(Unwrite(mean_label),Unwrite(std_label),Uncreate(axes),Uncreate(normal_curve),Uncreate(mean_line))

        # --------------------------------------------------------------------------------------------

        head = Text("Displaying the distribution of features")
        feature_one = features[:,[0]].tolist()
        feature_two = features[:,[1]].tolist()
        labels = labels.tolist()
        axes_1 = Axes(
            x_range=[min(feature_one)[0], max(feature_one)[0], 1],  # x轴范围：从-5到5，步长为1
            y_range=[min(labels)[0], max(labels)[0], 5],  # y轴范围：从-3到3，步长为1
            x_length=5,  # x轴的长度
            y_length=5,  # y轴的长度
            axis_config={"color": BLUE},  # 坐标轴的颜色
        )
        axes_2 = Axes(
            x_range=[min(feature_two)[0], max(feature_two)[0], 1],  # x轴范围：从-5到5，步长为1
            y_range=[min(labels)[0], max(labels)[0], 5],  # y轴范围：从-3到3，步长为1
            x_length=5,  # x轴的长度
            y_length=5,  # y轴的长度
            axis_config={"color": BLUE},  # 坐标轴的颜色
        )
        axes = VGroup(axes_1,axes_2)
        axes.arrange(RIGHT,buff=1)
        self.play(Create(axes))

        points_1 = []
        for i in range(len(labels)):
            dot_position = axes_1.coords_to_point(feature_one[i][0], labels[i][0])
            points_1.append(Dot(point=dot_position, radius=0.05, color=YELLOW))
        animations_1 = [Create(dot) for dot in points_1]
        points_2 = []
        for i in range(len(labels)):
            dot_position = axes_2.coords_to_point(feature_two[i][0], labels[i][0])
            points_2.append(Dot(point=dot_position, radius=0.05, color=YELLOW))
        animations_2 = [Create(dot) for dot in points_2]
        self.play(Succession(*animations_1, lag_ratio=0.005),Succession(*animations_2, lag_ratio=0.005))

        # 抽取样本-------------------------------------------------------------------------------------------- 

        head = Text('Shuffle the data and divided into samples(batches)',font_size=30)
        self.play(Uncreate(axes),Write(head),Uncreate(VGroup(*points_1)),Uncreate(VGroup(*points_2)))
        head.move_to(UP*3.5)
        code_text = '''
        def data_iter(batch_size,features,labels):
            num = len(features)
            index = list(range(num))
            random.shuffle(index)
            for i in range(0,num,batch_size):
                batch_index = torch.tensor(index[i:min(i+batch_size,num)])
                yield features[batch_index], labels[batch_index]
        '''
        code = Code(code=code_text,insert_line_no=False,language="Python",font="Monospace")
        self.play(Write(code))
        self.wait(2)
        self.play(Uncreate(code),Unwrite(head))

        def data_iter(batch_size,features,labels):
            num = len(features)
            index = list(range(num))
            random.shuffle(index)
            for i in range(0,num,batch_size):
                batch_index = torch.tensor(index[i:min(i+batch_size,num)])
                return batch_index.tolist()



        axes_1 = Axes(
            x_range=[min(feature_one)[0], max(feature_one)[0], 1],  # x轴范围：从-5到5，步长为1
            y_range=[min(labels)[0], max(labels)[0], 5],  # y轴范围：从-3到3，步长为1
            x_length=2.5,  # x轴的长度
            y_length=2.5,  # y轴的长度
            axis_config={"color": BLUE},  # 坐标轴的颜色
        )
        

        axes_2 = Axes(
            x_range=[min(feature_one)[0], max(feature_one)[0], 1],  # x轴范围：从-5到5，步长为1
            y_range=[min(labels)[0], max(labels)[0], 5],  # y轴范围：从-3到3，步长为1
            x_length=2.5,  # x轴的长度
            y_length=2.5,  # y轴的长度
            axis_config={"color": BLUE},  # 坐标轴的颜色
        )
        
        axes_3 = Axes(
            x_range=[min(feature_one)[0], max(feature_one)[0], 1],  # x轴范围：从-5到5，步长为1
            y_range=[min(labels)[0], max(labels)[0], 5],  # y轴范围：从-3到3，步长为1
            x_length=2.5,  # x轴的长度
            y_length=2.5,  # y轴的长度
            axis_config={"color": BLUE},  # 坐标轴的颜色
        )
        axes_4 = Axes(
            x_range=[min(feature_one)[0], max(feature_one)[0], 1],  # x轴范围：从-5到5，步长为1
            y_range=[min(labels)[0], max(labels)[0], 5],  # y轴范围：从-3到3，步长为1
            x_length=2.5,  # x轴的长度
            y_length=2.5,  # y轴的长度
            axis_config={"color": BLUE},  # 坐标轴的颜色
        )
        axes_5 = Axes(
            x_range=[min(feature_one)[0], max(feature_one)[0], 1],  # x轴范围：从-5到5，步长为1
            y_range=[min(labels)[0], max(labels)[0], 5],  # y轴范围：从-3到3，步长为1
            x_length=2.5,  # x轴的长度
            y_length=2.5,  # y轴的长度
            axis_config={"color": BLUE},  # 坐标轴的颜色
        )
        axes = VGroup(axes_1,axes_2,axes_3,axes_4,axes_5)
        axes.arrange(RIGHT)

        sample_1 = data_iter(10,features,labels)
        points_1 = []
        for i in sample_1:
            dot_position = axes_1.coords_to_point(features[i][0],labels[i][0])
            points_1.append(Dot(point=dot_position, radius=0.05, color=YELLOW))
        animations_1 = [Create(dot) for dot in points_1]

        sample_2 = data_iter(10,features,labels)
        points_2 = []
        for i in sample_2:
            dot_position = axes_2.coords_to_point(features[i][0],labels[i][0])
            points_2.append(Dot(point=dot_position, radius=0.05, color=YELLOW))
        animations_2 = [Create(dot) for dot in points_2]

        sample_3 = data_iter(10,features,labels)
        points_3 = []
        for i in sample_3:
            dot_position = axes_3.coords_to_point(features[i][0],labels[i][0])
            points_3.append(Dot(point=dot_position, radius=0.05, color=YELLOW))
        animations_3 = [Create(dot) for dot in points_3]

        sample_4 = data_iter(10,features,labels)
        points_4 = []
        for i in sample_4:
            dot_position = axes_4.coords_to_point(features[i][0],labels[i][0])
            points_4.append(Dot(point=dot_position, radius=0.05, color=YELLOW))
        animations_4 = [Create(dot) for dot in points_4]

        sample_5 = data_iter(10,features,labels)
        points_5 = []
        for i in sample_5:
            dot_position = axes_5.coords_to_point(features[i][0],labels[i][0])
            points_5.append(Dot(point=dot_position, radius=0.05, color=YELLOW))
        animations_5 = [Create(dot) for dot in points_5]   
        head = Text("Display five of Sample Batches (Batch Size = 10)",font_size=30)
        head.set_color(BLUE)
        head.move_to(UP*2.5)
        self.play(Write(head))
        self.play(Create(axes),Succession(*animations_1, lag_ratio=0.05),Succession(*animations_2, lag_ratio=0.05),Succession(*animations_3, lag_ratio=0.05),Succession(*animations_4, lag_ratio=0.05),Succession(*animations_5, lag_ratio=0.05))
        self.wait(3)
        self.play(Uncreate(axes),Uncreate(head),Uncreate(VGroup(*points_1)),Uncreate(VGroup(*points_2)),Uncreate(VGroup(*points_3)),Uncreate(VGroup(*points_4)),Uncreate(VGroup(*points_5)))

        # 定义模型-------------------------------------------------------------------------------------------- 

        head_1 = Text('Define the Function')
        head_1.set_color(BLUE)
        code_text_1 = '''
        def linreg(X, w, b):
            return torch.matmul(X, w) + b
        '''
        code_1 = Code(code=code_text_1,insert_line_no=False,language="Python",font="Monospace")
        head_2 = Text('Define the Loss Function')
        head_2.set_color(BLUE)
        code_text_2 = '''
        def squared_loss(y_hat, y):
            return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
        '''


        code_2 = Code(code=code_text_2,insert_line_no=False,language="Python",font="Monospace")
        head = VGroup(head_1,code_1,head_2,code_2)
        head.arrange(DOWN,buff=1)
        self.play(Write(head))
        self.wait(2)
        self.play(Unwrite(head))

        # 展示MSE--------------------------------------------------------------------------------------------    
        head = MathTex(r"Lose~Function~MSE~:(y_i - \hat{y}_i)^2")
        head.set_color(BLUE)
        head.move_to(UP*3)
        self.play(Write(head))
        axes = Axes(
            x_range=[-10, 10, 2.5],
            y_range=[0, 100, 20],
            x_length=10,
            y_length=5,
            axis_config={"color": GREEN},
        )
        
        # 定义MSE函数
        mse_curve = axes.plot(lambda x: (x**2), color=BLUE, x_range=[-10, 10])
        mse_der = axes.plot(lambda x: (2*x), color=RED, x_range=[-10, 10])
        # 将元素添加到场景中
        self.play(Create(axes),Create(mse_curve))
        self.wait(2)
        self.play(Uncreate(head))
        head = Text("The MSE Derivative indicates that loss will be increasing as it increase",font_size=30)
        head.set_color(BLUE)
        head.move_to(UP*3)
        self.play(Write(head),Create(mse_der))
        self.wait(3)
        self.play(Uncreate(head),Uncreate(mse_der),Uncreate(mse_curve),Uncreate(axes))


        # 展示SGD--------------------------------------------------------------------------------------------
        head = Text("Now Conduct the Optimization Method")
        head.move_to(UP*3)
        head.set_color(BLUE)
        code_text = '''
        def sgd(params, lr, batch_size):
        with torch.no_grad():
            for param in params:
                param -= lr * param.grad / batch_size
                param.grad.zero_()
        '''
        code = Code(code=code_text,insert_line_no=False,language="Python",font="Monospace")
        head_2 = Text('Apply this optimization method for each batch')
        head_2.set_color(BLUE)
        sgd = MathTex(r"(w,b)\leftarrow (w,b)-\eta g")
        main = VGroup(head,code,head_2,sgd)
        main.arrange(DOWN,buff=0.7)
        self.play(Write(main))
        self.wait(2)
        self.play(Uncreate(main),run_time=0.1)
        
        # 计算梯度--------------------------------------------------------------------------------------------
        head = Text("Now Calculate the Gradient")
        head.set_color(BLUE)
        head.move_to(UP*3)
        grad = MathTex(r"\frac{\partial \text{MSE}}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} \frac{\partial}{\partial w} \left( y_i - (w x_i + b) \right)^2")
        grad_1 = MathTex(r"= \frac{1}{n} \sum_{i=1}^{n} 2(y_i - (wx_i + b)) \cdot (-x_i)")
        grad_2 = MathTex(r"= -\frac{1}{n} \sum_{i=1}^{n} 2(y_i - \hat{y}_i) \cdot x_i")

        main = VGroup(head,grad,grad_1,grad_2)
        main.arrange(DOWN,buff=0.7)
        self.play(Write(main))
        self.wait(2)
        self.play(Uncreate(main),run_time=0.01)
        head = Text("Then apllies the formula for 1000/10=100 Times",font_size=45)
        head.set_color(BLUE)
        grad = MathTex(r'w := w + \eta \cdot \frac{1}{n} \sum_{i=1}^{n} 2(y_i - \hat{y}_i) \cdot x_i')
        grad_1 = MathTex(r"b := b + \eta \cdot \frac{1}{n} \sum_{i=1}^{n} 2(y_i - \hat{y}_i)")
        main = VGroup(head,grad,grad_1)
        main.arrange(DOWN,buff=0.7)
        self.play(Write(main))
        self.wait(3)
        self.play(Uncreate(main),run_time=0.01)

        # 总结--------------------------------------------------------------------------------------------
        code_text = '''
        lr = 0.03
        num_epochs = 3
        net = linreg
        loss = squared_loss

        for epoch in range(num_epochs):
            for X, y in data_iter(batch_size, features, labels):
                l = loss(net(X, w, b), y)
                l.sum().backward()
                sgd([w, b], lr, batch_size)
            with torch.no_grad():
                train_l = loss(net(features, w, b), labels)
                print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        '''
        code = Code(code=code_text,insert_line_no=False,language="Python",font="Monospace")
        head = Text("Then applies the whole process in epochs and that's linear regression",font_size=30)
        main = VGroup(head,code)
        main.arrange(DOWN,buff=1)
        self.play(Write(main))