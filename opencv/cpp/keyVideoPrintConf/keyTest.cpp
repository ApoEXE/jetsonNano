//g++ keyTest.cpp -o test -pthread -lncurses

#include <iostream>
#include <unistd.h>
#include "conio.h"

int main()
{

    while (true)
    {
        if (kbhit() != 0)
        {
            std::cout << getch() << std::endl;
        }
    }
}