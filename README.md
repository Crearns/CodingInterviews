# 《剑指offer》题解
更新时间：2019/2/1
# content
* [1、二维数组中的查找](#二维数组中的查找)
* [2、从尾到头打印链表](#从尾到头打印链表)
* [3、重建二叉树](#重建二叉树)
* [4、用两个栈实现队列](#用两个栈实现队列)
* [5、旋转数组的最小数字](#旋转数组的最小数字)
* [6、斐波那契数列](#斐波那契数列)
* [7、跳台阶](#跳台阶)
* [8、变态跳台阶](#变态跳台阶)
* [9、矩形覆盖](#矩形覆盖)




# 正文
## 二维数组中的查找

### 题目描述
在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。

### 题解
```java
public class Solution {
    public boolean Find(int target, int [][] array) {
        int lo, lh, mid;
        if (array == null || array.length == 0 || array[0].length == 0)
            return false;
        for (int i = 0; i < array.length; i++) {
            if ((array[i][0] > target || array[i][array.length-1] < target))
                continue;
            lo = 0;
            lh = array.length - 1;
            while (lo <= lh){
                mid = lo + (lh-lo)/2;
                if (target < array[i][mid]){
                    lh = mid - 1;
                }
                else if (target > array[i][mid]){
                    lo = mid + 1;
                }
                else
                    return true;
            }
        }
        return false;
    }
}
```

## 从尾到头打印链表

### 题目描述
输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。

### 题解
LinkedList实现了stack接口，运用栈的后进先出特性，最后使用addAll方法返回arraylist
```java
/**
*    public class ListNode {
*        int val;
*        ListNode next = null;
*
*        ListNode(int val) {
*            this.val = val;
*        }
*    }
*
*/
import java.util.ArrayList;
import java.util.LinkedList;
public class Solution {
    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
        LinkedList<Integer> linkedList = new LinkedList<>();
        ArrayList<Integer> arrayList = new ArrayList<>();
        if (listNode == null)
            return arrayList;
        while (listNode != null){
            linkedList.push(listNode.val);
            listNode = listNode.next;
        }
        arrayList.addAll(linkedList);
        return arrayList;
    }
}
```

## 重建二叉树

### 题目描述
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。

### 题解
```java
/**
 * Definition for binary tree
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */

public class Solution {
    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {
        TreeNode root=reConstructBinaryTree(pre,0,pre.length-1,in,0,in.length-1);
        return root;
    }
    //前序遍历{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}
    private TreeNode reConstructBinaryTree(int [] pre,int startPre,int endPre,int [] in,int startIn,int endIn) {

        if(startPre>endPre||startIn>endIn)
            return null;
        TreeNode root=new TreeNode(pre[startPre]);
                                                                                                                                                                                                                                                                                                                                         
        for(int i=startIn;i<=endIn;i++)
            if(in[i]==pre[startPre]){
                root.left=reConstructBinaryTree(pre,startPre+1,startPre+i-startIn,in,startIn,i-1);
                root.right=reConstructBinaryTree(pre,i-startIn+startPre+1,endPre,in,i+1,endIn);
            }

        return root;
    }                                           
}
```

## 用两个栈实现队列

### 题目描述
用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。

### 题解

```java
import java.util.Stack;

public class Solution {
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();
    
    public void push(int node) {
        stack1.push(node);
    }

    public int pop() {
        while(!stack1.isEmpty()){
            stack2.push(stack1.pop());
        }
        int first=stack2.pop();
        while(!stack2.isEmpty()){
            stack1.push(stack2.pop());
        }
        return first;
    }
}
```

## 旋转数组的最小数字
### 题目描述
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

### 题解
使用二分查找法

```java
import java.util.ArrayList;
import java.util.TreeSet;
public class Solution {
    public int minNumberInRotateArray(int [] array) {
        if (array.length == 0)
            return 0;
        int left = 0;
        int right = array.length - 1;
        int middle = -1;
        while (array[left]>=array[right]) {
            if(right-left==1){
                middle = right;
                break;
            }
            middle = left + (right - left) / 2;
            if (array[middle] >= array[left]) {
                left = middle;
            }
            if (array[middle] <= array[right]) {
                right = middle;
            }
        }
        return array[middle];
    }
}
```

## 斐波那契数列
### 题目描述
大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
n<=39

### 题解

```java
public class Solution {
    public int Fibonacci(int n) {
        if (n == 0)
            return 0;
        else if (n == 1)
            return 1;
        int pre = 0, curr = 1, next = 0;
        for (int i = 1; i < n; i++) {
            next = curr + pre;
            pre = curr;
            curr = next;
        }
        return next;
    }
}
```


## 跳台阶

### 题目描述
一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

### 题解
此题为斐波那契数列的变形。如果只有1级台阶，则只有一种跳法；如果有2级台阶，则可以一次跳两阶，或者一次跳一阶；当n>2时，当n青蛙在最后一跳的时候有两种情况：1、跳一级到达终点；2、跳两级到达终点。则可以得到
```
f(n) = 1                (n = 1)
f(n) = 2                (n = 2)
f(n) = f(n-1) + f(n-2)  (n > 2) 
```

```java
public class Solution {
    public int JumpFloor(int target) {
      if (target == 1)
            return 1;
        if (target == 2)
            return 2;
        int fn1=1,fn2=2,sum=0;
        while(target>2){
            sum =fn1+fn2;
            fn1=fn2;
            fn2=sum;
            target--;
        }
        return sum;
    }
}
```


## 变态跳台阶

### 题目描述
一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

### 题解
此题为斐波那契数列的变形，与上题类似，思路仍然为第一次跳多少台阶，则跳法数等于剩下多少台阶的跳法数目，与上题不同的是一开始就可以一次跳完，所以f(n)=1
```
如果有1级台阶，则有f(1) = 1 种跳法
如果有2级台阶，则有f(2) = f(2-1) + f(2-2) = 2 种跳法
如果有3级台阶，则有f(3) = f(3-1) + f(3-2) + f(3-3) = 4 种跳法
···
如果有n级台阶，则有f(n) = f(n-1) + f(n-2) + f(n-3) + ··· + f(0) 种跳法
              又 f(n-1) = f(n-2) + f(n-3) + f(n-4) + ··· + f(0)
     进行相减可得，f(n) - f(n-1) = f(n-1)
              即，f(n) = 2f(n-1)
              
由此得出，
f(n) = 1, 当n=0时
f(n) = 1, 当n=1时
f(n) = f(n-1) + f(n-2), 当n>=2时
```

```java
public class Solution {
    public int JumpFloorII(int n) {
        if(n <= 0)
            return 0;
        if(n == 1)
            return 1;
        int num = 0;
        int num1 = 1;   //初始值应为1而非0
        int num2 = 1;
        for(int i = 2; i <= n; i++){
            num = num1 + num2;
            num2 = num1;
            num1 = num;
        }
        return num;
    }
}
```

### 题解2

可以归纳前面几项得到规律可以得到f(n) = 2^(n-1)

```java
public class Solution {
    public static int JumpFloorII(int target) {
        return 1 << target-1;
    }
}

```

## 矩形覆盖

### 题目描述
我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

### 题解
此题是斐波那契数列的变形

![](https://i.loli.net/2019/02/01/5c53b58f9c3b3.png)
图源牛客网follow


```java
public class Solution {
    public int RectCover(int n) {
        if(n <= 0)
            return 0;
        if(n == 1)
            return 1;
        int num = 0;
        int num1 = 1;
        int num2 = 1;
        for(int i = 2; i <= n; i++){
            num = num1 + num2;
            num2 = num1;
            num1 = num;
        }
        return num;
    }
}

```