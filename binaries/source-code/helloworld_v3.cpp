// helloworld_v3.cpp
#include <iostream>
#include <string>

// 1) Greeting adds a punctuation change
void greet(const std::string &name) {
    std::cout << "Hello, " << name << "!!" << std::endl;  // two exclamation marks
}

// 2) Add now prints the sum before returning it
int add(int a, int b) {
    int result = a + b;
    std::cout << "[Debug] a + b = " << result << std::endl;
    return result;
}

// 3) Multiply uses a loop instead of direct multiplication
int multiply(int a, int b) {
    int product = 0;
    for (int i = 0; i < b; ++i) {
        product += a;
    }
    return product;
}

int main() {
    greet("Alice");

    int x = 3, y = 5;
    std::cout << "add(" << x << ", " << y << ") = " << add(x, y) << std::endl;
    std::cout << "multiply(" << x << ", " << y << ") = " << multiply(x, y) << std::endl;

    return 0;
}