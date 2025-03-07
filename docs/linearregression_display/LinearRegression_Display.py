from manim import *
import numpy as np
import torch
import random

class LinearRegression(Scene):
    def construct(self):
        def data_generator(w, b, num):
            X = torch.normal(0, 1, (num, len(w)))
            y = torch.matmul(X, w) + b
            y += torch.normal(0, 0.01, y.shape)
            return X, y.reshape((-1, 1))

        true_w = torch.tensor([2, -3.4])
        true_b = 4.2
        features, labels = data_generator(true_w, true_b, 1000)

        head = Text("Linear Regression Display - Buezwqwg")
        head.set_color(BLUE)
        self.play(Create(head))
        self.wait(1)
        self.play(Uncreate(head))

        feature_one = features[:, [0]].tolist()
        feature_two = features[:, [1]].tolist()
        labels_list = labels.tolist()
        axes_1 = Axes(
            x_range=[min(feature_one)[0]-1, max(feature_one)[0]+1, 1],
            y_range=[min(labels_list)[0]-1, max(labels_list)[0]+1, 5],
            x_length=5,
            y_length=5,
            axis_config={"color": BLUE},
        )
        axes_2 = Axes(
            x_range=[min(feature_two)[0]-1, max(feature_two)[0]+1, 1],
            y_range=[min(labels_list)[0]-1, max(labels_list)[0]+1, 5],
            x_length=5,
            y_length=5,
            axis_config={"color": BLUE},
        )
        axes = VGroup(axes_1, axes_2)
        axes.arrange(RIGHT, buff=1)
        self.play(Create(axes))

        points_1 = []
        for i in range(len(labels_list)):
            dot_position = axes_1.coords_to_point(feature_one[i][0], labels[i][0])
            points_1.append(Dot(point=dot_position, radius=0.03, color=YELLOW))
        points_2 = []
        for i in range(len(labels_list)):
            dot_position = axes_2.coords_to_point(feature_two[i][0], labels[i][0])
            points_2.append(Dot(point=dot_position, radius=0.03, color=YELLOW))
        points_group_1 = VGroup(*points_1)
        points_group_2 = VGroup(*points_2)
        self.play(Create(points_group_1), Create(points_group_2))

        # Draw the true model lines
        true_line_1 = axes_1.plot(lambda x: true_w[0].item() * x + true_b, color=GREEN)
        true_line_2 = axes_2.plot(lambda x: true_w[1].item() * x + true_b, color=GREEN)
        self.play(Create(true_line_1), Create(true_line_2))

        # Initialize parameters
        w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        w_1 = w.detach().numpy()[0][0]
        w_2 = w.detach().numpy()[1][0]
        b_0 = b.detach().numpy()[0]
        line_1 = axes_1.plot(lambda x: w_1 * x + b_0, color=BLUE)
        line_2 = axes_2.plot(lambda x: w_2 * x + b_0, color=BLUE)
        self.play(Create(line_1), Create(line_2))

        # Add text to display loss, w, b
        loss_text = MathTex(f"\\text{{Loss}} = {0:.4f}")
        loss_text.to_edge(UP)
        w_text = MathTex(f"w_1 = {w_1:.4f},\\ w_2 = {w_2:.4f}")
        w_text.next_to(loss_text, DOWN)
        b_text = MathTex(f"b = {b_0:.4f}")
        b_text.next_to(w_text, DOWN)
        param_text = VGroup(loss_text, w_text, b_text)
        self.play(Write(param_text))

        def data_iter(batch_size, features, labels):
            num = len(features)
            index = list(range(num))
            random.shuffle(index)
            for i in range(0, num, batch_size):
                batch_index = torch.tensor(index[i:min(i+batch_size, num)])
                yield features[batch_index], labels[batch_index]

        batch_size = 10

        def linreg(X, w, b):
            return torch.matmul(X, w) + b

        def squared_loss(y_hat, y):
            return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

        def sgd(params, lr, batch_size):
            with torch.no_grad():
                for param in params:
                    param -= lr * param.grad / batch_size
                    param.grad.zero_()

        lr = 0.03
        num_epochs = 3  # Changed to 3 epochs
        net = linreg
        loss = squared_loss

        update_interval = 1  # Update after every batch
        batch_counter = 0

        # Start training and update after every backpropagation
        for epoch in range(num_epochs):
            for X, y in data_iter(batch_size, features, labels):
                l = loss(net(X, w, b), y)
                l.sum().backward()
                sgd([w, b], lr, batch_size)

                batch_counter += 1
                # Get the latest parameters
                new_w_1 = w.detach().numpy()[0][0]
                new_w_2 = w.detach().numpy()[1][0]
                new_b_0 = b.detach().numpy()[0]

                # Update lines
                new_line_1 = axes_1.plot(lambda x: new_w_1 * x + new_b_0, color=BLUE)
                new_line_2 = axes_2.plot(lambda x: new_w_2 * x + new_b_0, color=BLUE)
                self.play(
                    Transform(line_1, new_line_1),
                    Transform(line_2, new_line_2),
                    run_time=0.1  # Adjusted animation time for smoother update
                )

                # Update loss and parameter displays
                with torch.no_grad():
                    train_l = loss(net(features, w, b), labels)
                    current_loss = float(train_l.mean())
                new_loss_text = MathTex(f"\\text{{Loss}} = {current_loss:.4f}")
                new_loss_text.to_edge(UP)
                new_w_text = MathTex(f"w_1 = {new_w_1:.4f},\\ w_2 = {new_w_2:.4f}")
                new_w_text.next_to(new_loss_text, DOWN)
                new_b_text = MathTex(f"b = {new_b_0:.4f}")
                new_b_text.next_to(w_text, DOWN)
                self.play(
                    Transform(loss_text, new_loss_text),
                    Transform(w_text, new_w_text),
                    Transform(b_text, new_b_text),
                    run_time=0.1
                )

            # Output current epoch's loss
            print(f'epoch {epoch + 1}, loss {current_loss:f}')

        self.wait(2)
