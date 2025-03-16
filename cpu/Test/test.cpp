/*
记录学习C++八股过程中的一些测试文件
*/

#include <iostream>
using namespace std;


// class A
// {
// public:
//     static const int a = 10;
//     A(int x){};
//     A(){};
// };

struct A
{
public:
    int a = 10;
};

int main()
{
    A x;
    x.a;
}