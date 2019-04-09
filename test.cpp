#include <iostream>
using namespace std;

int count2index(int temp)
{
    int ret = 0 ;
    while((temp % 2) != 1)
    {
        ret++;
        temp = temp / 2;
    }
    return ret;
}

int mian()
{
    int n = 0;
    int m[100000] = {0};
    int temp = 0;
    cin >> n;
    
    // 计算指数集合
    for(int i = 0; i < n ; i++)
    {
      cin >> temp;
      m[i] = count2index(temp);
    }
    
    // 排序指数集
    for(int i = 1; i < n ; i++)
    {
        for(int j = 0; j < n - i - 1; j++)
        {
            if(m[j] > m[j+1])
            {
                temp = m[j]
            
            }
        }
    }
    
  
    return 0;
}
