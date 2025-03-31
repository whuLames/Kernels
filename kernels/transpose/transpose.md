## Transpose Kernel 实现记录

### v1: 朴素版本实现
朴素版本的实现最好理解, 即对于输入矩阵的坐标(i,j), 将其元素写入转置矩阵位置(j,i)的元素中
1. 对于输入的矩阵A=M*N, 我们在设置blocksize和gridsize时, 不管X维度是切分M还是切分N,都不能保证在数据读取和写入时都保证合并读写
2. 所谓合并读写即为连续的线程读写连续的内存, 具体来讲,每个warp内的线程, 其X方向是需要连续读写的

### v2: 合并读取
合并读取的优化即通过shared_memory的作用, 保证读取和写入都为连续的读写
对于shared_memory来说，其先读取矩阵A的一块小矩阵，分为两种情况:
- 连续读取,即按行读取A中的小矩阵, 并且按行写入共享内存。然后按列读取小矩阵并按行写入B矩阵
- 按行读取A中小矩阵, 并按列写入共享内存。然后按行读取小矩阵, 并按行写入B矩阵

这里我们可以看到使用共享内存的本质原因: 共享内存非连续读取的代价小于全局内存的非连续读取。
我们用共享内存的非连续读取来换取全局内存的连续读取。其本质依然是个tradeoff。
只不过其他情况下共享内存的优点在于多次读取时节省速度。

### v3：padding 去除 bank conflict

### v4：swizzling 去除 bank conflict


### bank conflict: 
- 不同的线程访问同一bank的不同address时就会出现bank conflict
- bank conflict只发生在同一个warp的不同线程间。(以warp为单位进行调度，即一个指令发射的作用域为一个warp)
 - 一个block中的不同warp是否可以同时执行？
- 如果多个线程访问shared memory的相同bank的相同address，实际效果是broadcast，非bank conflict。
- bank conflict只发生在shared memory的读写操作上，global memory的读写操作不会有bank conflict产生。