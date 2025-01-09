from manim import *
import random
import time
import numpy as np

class CoordinateSystem(Scene):
    def construct(self):
        # 参数设置
        k = 1  # 空气阻力系数
        m = 1  # 物体质量
        g = 10  # 重力加速度
        v0x = 10  # 水平初速度
        v0y = 10  # 垂直初速度

        # 创建一个坐标系
        axes = Axes(
            x_range=[0, 10, 1],  # x轴范围：从-5到5，步长为1
            y_range=[0, 10, 1],  # y轴范围：从-3到3，步长为1
            x_length=5,  # x轴的长度
            y_length=5,  # y轴的长度
            axis_config={"color": BLUE},  # 坐标轴的颜色
            tips=True # 是否显示坐标轴的箭头
        )

        start_point = axes.coords_to_point(0, 0)
        end_point = axes.coords_to_point(7, 7)
        vector = Vector().put_start_and_end_on(start_point, end_point).set_color(GREEN) 
        vector.set_stroke(width=4)
        #angle = Angle(basis_x,basis_y)

        # 为坐标系添加标签
        labels = axes.get_axis_labels(x_label="x", y_label="y")
        head = Text("Projectile Motion (Ff=-v) - Buezwqwg",font_size=30)
        head.set_color(BLUE)
        head.next_to(labels,UP)
        head.set_color(BLUE)
        direction_vector = Text("u=[1,1]",font_size=30)
        direction_vector.next_to(axes)
        direction_vector.set_color(GREEN)
        direction_vector.next_to(vector,RIGHT)
        # 显示坐标系和标签
        main = VGroup(axes,labels,vector,direction_vector)
        main.move_to(ORIGIN)


        self.play(Write(head),run_time=1)
        self.play(Write(main),run_time=1)
        self.wait(1)

#-------------------------------------------------------------------------------------------------------------

        self.play(Uncreate(head),main.animate.move_to(LEFT*3),run_time=0.5)

        head_1 = Text("For a object projectile along direction u=[1,1]\nIt will have the same initial velocity on x and y direction",font_size=30)
        head_1.move_to([2,2.5,0])
        head_1.set_color(BLUE)
        self.play(Write(head_1),run_time=1)
        vector_vx = Vector([0,2,0]).shift(vector.get_start())
        vector_vy = Vector([2,0,0]).shift(vector.get_start())
        vector_vx.set_color(GREEN)
        vector_vy.set_color(GREEN)
        head_vx = Text("v0x",font_size=30)
        head_vx.set_color(GREEN)
        head_vy = Text("v0y",font_size=30)
        head_vy.set_color(GREEN)
        head_vx.next_to(vector_vx,LEFT)
        head_vy.next_to(vector_vy,DOWN)
        self.play(Write(vector_vx),Write(vector_vy),Write(head_vx),Write(head_vy),run_time=0.5)
        self.wait(2)
        main = VGroup(main,vector_vx,vector_vy,head_vx,head_vy)
        main_position = main.get_center()
        self.play(Uncreate(head_1),main.animate.move_to([-15,0,0]),run_time=1)
        
#-------------------------------------------------------------------------------------------------------------

        #latex x direction
        head = Text("Thus on the x direction we can have",font_size=20)
        latex_x_1 = MathTex(r"F=ma=-kv_x",font_size=30)
        latex_x_2 = MathTex(r"\frac{dv_x}{dt}=-\frac{k}{m}v_x",font_size=30)
        latex_x_3 = Text("Use the Speration of Variable to get",font_size=20)
        latex_x_4 = MathTex(r"\int \frac{dv_x}{v_x}=\int-\frac{k}{m}dt",font_size=30)
        latex_x_5 = MathTex(r"v_x=e^{-\frac{k}{m}}\cdot C",font_size=30)
        latex_x_6 = Text("Consider the Particular Solution",font_size=20)
        latex_x_7 = MathTex(r"t=0,v_{0x}=v_{0x \Rightarrow v_{0x}=C",font_size=30)
        latex_x_8 = Text("So got the Particular Solution of Differential Equation for x-Axis",font_size=20)
        latex_x_9 = MathTex(r"v_x=v_{0x}\cdot e^{-\frac{k}{m}}",font_size=30)
        #latex_x.next_to(head,DOWN)
        latex_x = VGroup(head,latex_x_1,latex_x_2,latex_x_3,latex_x_4,latex_x_5,latex_x_6,latex_x_7,latex_x_8,latex_x_9)
        latex_x.arrange(DOWN)
        latex_x.set_color(BLUE)
        self.play(Write(latex_x),run_time=5)
        self.wait(1)
        self.play(Unwrite(latex_x),run_time=0.5)

#-------------------------------------------------------------------------------------------------------------
        # latex y direction 
        head = Text("On the y-axis is the same",font_size=30)
        latex_y_1 = MathTex(r"F=ma=-mg-kv_y",font_size=30)
        latex_y_2 = MathTex(r"m\frac{dv_{y}}{dt}=-mg-kv_y",font_size=30)
        latex_y_3 = MathTex(r"v'+\frac{k}{m}v_y=-g",font_size=30)
        latex_y_4 = Text("Based on FODE, come out with the solution",font_size=20)
        latex_y_5 = MathTex(r"v_y=e^{-\frac{k}{m}t}(\int-ge^{\frac{k}{m}t}dt+C)",font_size=30)
        latex_y_6 = MathTex(r"v_y=e^{-\frac{k}{m}t}(-g \frac{m}{k}e^{\frac{k}{m}t}+C)",font_size=30)
        latex_y_7 = Text("Find Particular Solution",font_size=20)
        latex_y_8 = MathTex(r"when~t=0,~v_{0y}=v_{0y}",font_size=20)
        latex_y_9 = MathTex(r"v_{0y}=-\frac{mg}{k}+C\Rightarrow C=v_{0y}+\frac{mg}{k}",font_size=30)
        latex_y_10 = Text("So got the Particular Solution for y-Axis",font_size=20)
        latex_y_11 = MathTex(r"v_y(t)=\frac{-mg}{k}+(v_{0y}+\frac{mg}{k})e^{-\frac{k}{m}t}",font_size=30)

        #latex_x.next_to(head,DOWN)
        latex_y = VGroup(head,latex_y_1,latex_y_2,latex_y_3,latex_y_4,latex_y_5,latex_y_6,latex_y_8,latex_y_9,latex_y_10,latex_y_11)
        latex_y.arrange(DOWN)
        latex_y.set_color(BLUE)
        self.play(Write(latex_y),run_time=5)
        self.wait(1)
        self.play(Unwrite(latex_y),run_time=0.5)

#-------------------------------------------------------------------------------------------------------------

        #Eliminate to Parametric Equation
        head = Text("Eliminating the Parameter of Parametric Equation",font_size=30)
        latex_e_1 = Text("Eliminate t through y",font_size=20)
        latex_e_2 = MathTex(r"x(t)=\frac{mv_{0x}}{k}(1-e^{-\frac{k}{m}t})",font_size=30)
        latex_e_3 = MathTex(r"e^{-\frac{m}{k}t}=1-\frac{kx}{mv_{0x}}",font_size=30)
        latex_e_4 = MathTex(r"t=-\frac{m}{k}\ln(1-\frac{kx}{mv_{0x}})",font_size=30)

        latex_e_5 = Text("Bring t into y",font_size=20)
        latex_e_6 = MathTex(r"y(t)=-\frac{mg}{k}(-\frac{m}{k}\ln(1-\frac{kx}{mv_{0x}})))+(v_{0y}+\frac{mg}{k})\frac{m}{k}(1-e^{-\frac{k}{m}(-\frac{m}{k}\ln(1-\frac{kx}{mv_{0x}}))})",font_size=30)
        latex_e_7 = MathTex(r"y=\frac{mg}{k^2}m\ln(1-\frac{kx}{mv_{0x}})+(v_{0y+\frac{mg}{k}})\frac{m}{k}\frac{kx}{mv_{0x}}",font_size=30)
        latex_e_8 = MathTex(r"\Rightarrow y=\frac{m^2g}{k^2}ln(1- \frac{kx}{mv_{0x}})+\frac{v_{0y}}{v_{0x}}x+\frac{mg}{kv_{0x}}x",font_size=30)
        #latex_x.next_to(head,DOWN)
        
        latex_x = VGroup(head,latex_e_1,latex_e_2,latex_e_3,latex_e_4,latex_e_5,latex_e_6,latex_e_7,latex_e_8)
        latex_x.arrange(DOWN)
        latex_x.set_color(BLUE)
        latex_e_8.set_color(RED)
        self.play(Write(latex_x),run_time=5)
        self.wait(1)
        self.play(Unwrite(latex_x),run_time=0.5)
        self.play(main.animate.move_to(ORIGIN))
#-------------------------------------------------------------------------------------------------------------

        def parafunc_no_k(t):
            x = v0x * t  # 水平方向运动
            y = v0y * t - 0.5 * g * t**2  # 垂直方向运动
            return np.array([x, y, 0])

        
        def parametric_curve(t):
            x = (m*v0x)/k*(1-np.exp(-k*t/m))  # 水平方向运动
            y = -(m*g*t)/k+(v0y+m*g/k)*m/k*(1-np.exp(-k*t/m))
            return np.array([x, y, 0])

        # 绘制参数曲线

        trajectory = ParametricFunction(
            lambda t: axes.coords_to_point(*parametric_curve(t)),
            t_range=[0, 5],  # t的范围
            color=GREEN
        )



        self.play(Write(trajectory),run_time=4)



        self.wait(1)


