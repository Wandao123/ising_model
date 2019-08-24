//#include <iostream>
#include <cstdlib>
#include "ising_model.h"

IsingModel isingModel(20.e0);
static bool IsUpdating = false;

void idle(unsigned long int countedFrames)
{
	if (IsUpdating)
		isingModel.Update();
}

void display(GLFWwindow* window)
{
	glClearColor(1.0f, 1.0f, 1.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	isingModel.Draw();
	glfwSwapBuffers(window);
	//glFlush();
}

void mouseButton(GLFWwindow* window, int button, int action, int mods)
{
	if (action == GLFW_PRESS) {
		switch (button) {
		case GLFW_MOUSE_BUTTON_LEFT:
		case GLFW_MOUSE_BUTTON_MIDDLE:
		case GLFW_MOUSE_BUTTON_RIGHT:
			isingModel.Update();
			break;
		}
	}
}

void keyboard(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (action == GLFW_RELEASE) {
		switch (key) {
		case GLFW_KEY_Q:
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, GL_TRUE);
		}		
	} else {
		switch (key) {
		case GLFW_KEY_A:
			isingModel.SwitchAutoCooling();
			break;
		case GLFW_KEY_C:
			isingModel.ChangeAlgorithm();
			break;
		case GLFW_KEY_SPACE:
			if (IsUpdating) {
				IsUpdating = false;
			} else {
				IsUpdating = true;
			}
			break;
		case GLFW_KEY_UP:
			isingModel.Increase();
			break;
		case GLFW_KEY_DOWN:
			isingModel.Decrease();
			break;
		}
	}
}

int main(int argc, char *argv[])
{
	// Initialization
	if (!glfwInit())
		return -1;
	//glfwWindowHint(GLFW_DOUBLEBUFFER, GL_FALSE);
	GLFWwindow* window = glfwCreateWindow(ScreenWidth, ScreenHeight, "Ising model", nullptr, nullptr);
	if (!window) {
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, ScreenWidth, ScreenHeight, 0.0, -1.0, 1.0);

	// Setting call back functions
	glfwSetMouseButtonCallback(window, mouseButton);
	glfwSetKeyCallback(window, keyboard);

	// Event loop
	unsigned long int frameCount = 0;
	double previousTime = glfwGetTime();  unsigned long int previousFrame = 0;
	while (!glfwWindowShouldClose(window)) {
		++frameCount;

		// Calculate FPS
		/*double currentTime = glfwGetTime();
		if (currentTime - previousTime >= 1.0) {
			std::cout << frameCount - previousFrame << std::endl;
			previousTime = currentTime;
			previousFrame = frameCount;
		}*/

		idle(frameCount);
		display(window);
		glfwPollEvents();
	}
	glfwTerminate();
	return 0;
}
