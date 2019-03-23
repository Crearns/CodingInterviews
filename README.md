# 《剑指offer》题解和笔记
更新时间：2019/3/22
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
* [10、二进制中1的个数](#二进制中1的个数)
* [11、数值的整数次方](#数值的整数次方)
* [12、调整数组顺序使奇数位于偶数前面](#调整数组顺序使奇数位于偶数前面)
* [13、链表中倒数第k个结点](#链表中倒数第k个结点)
* [14、反转链表](#反转链表)
* [15、合并两个排序的链表](#合并两个排序的链表)
* [16、树的子结构](#树的子结构)
* [17、二叉树的镜像](#二叉树的镜像 )
* [18、顺时针打印矩阵](#顺时针打印矩阵)
* [19、包含min函数的栈](#包含min函数的栈)
* [20、栈的压入、弹出序列](#栈的压入、弹出序列)
* [21、从上往下打印二叉树](#从上往下打印二叉树)
* [22、二叉搜索树的后序遍历序列](#二叉搜索树的后序遍历序列)
* [23、二叉树中和为某一值的路径](#二叉树中和为某一值的路径)
* [24、复杂链表的复制](#复杂链表的复制)
* [25、二叉搜索树与双向链表](#二叉搜索树与双向链表)
* [26、字符串的排列](#字符串的排列)
* [27、数组中出现次数超过一半的数字](#数组中出现次数超过一半的数字)
* [28、最小的K个数](#最小的K个数)
* [29、连续子数组的最大和](#连续子数组的最大和)
* [30、整数中1出现的次数（从1到n整数中1出现的次数）](#整数中1出现的次数（从1到n整数中1出现的次数）)
* [31、把数组排成最小的数](#把数组排成最小的数)
* [32、丑数](#丑数)
* [33、第一个只出现一次的字符](#第一个只出现一次的字符)
* [34、数组中的逆序对](#数组中的逆序对)
* [35、两个链表的第一个公共结点](#两个链表的第一个公共结点)
* [36、数字在排序数组中出现的次数](#数字在排序数组中出现的次数)
* [37、二叉树的深度](#二叉树的深度)
* [38、平衡二叉树](#平衡二叉树)
* [39、数组中只出现一次的数字](#数组中只出现一次的数字)
* [40、和为S的连续正数序列](#和为S的连续正数序列)
* [41、和为S的两个数字](#和为S的两个数字)
* [42、左旋转字符串](#左旋转字符串)
* [43、翻转单词顺序列](#翻转单词顺序列)
* [44、扑克牌顺子](#扑克牌顺子)
* [45、孩子们的游戏(圆圈中最后剩下的数)](#孩子们的游戏圆圈中最后剩下的数)
* [46、求1+2+3+...+n](#https://github.com/Crearns/CodingInterviews#%E6%B1%82123n)
* [47、不用加减乘除做加法](#不用加减乘除做加法)
* [48、把字符串转换成整数](#把字符串转换成整数)
* [49、数组中重复的数字](#数组中重复的数字)
* [50、构建乘积数组](#构建乘积数组)
* [51、正则表达式匹配](#正则表达式匹配)
* [52、表示数值的字符串](#表示数值的字符串)
* [53、字符流中第一个不重复的字符](#字符流中第一个不重复的字符)
* [54、链表中环的入口结点](#链表中环的入口结点)
* [55、删除链表中重复的结点](#删除链表中重复的结点)
* [56、二叉树的下一个结点](#二叉树的下一个结点)
* [57、对称的二叉树](#对称的二叉树)
* [58、按之字形顺序打印二叉树](#按之字形顺序打印二叉树)
* [59、把二叉树打印成多行](#把二叉树打印成多行)
* [60、序列化二叉树](#序列化二叉树)
* [61、二叉搜索树的第k个结点](#二叉搜索树的第k个结点)
* [62、数据流中的中位数](#数据流中的中位数)
* [63、滑动窗口的最大值](#滑动窗口的最大值 )
* [64、矩阵中的路径](#矩阵中的路径)
* [65、机器人的运动范围](#机器人的运动范围)

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

## 二进制中1的个数

### 题目描述
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

### 题解
#### n & (n-1)
该位运算去除 n 的位级表示中最低的那一位。
```
n       : 10110100
n-1     : 10110011
n&(n-1) : 10110000
```
时间复杂度：O(M)，其中 M 表示 1 的个数。



```java
public int NumberOf1(int n) {
    int cnt = 0;
    while (n != 0) {
        cnt++;
        n &= (n - 1);
    }
    return cnt;
}
```

## 数值的整数次方

### 题目描述
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

### 题解
#### 循环
```java
public class Solution {
    public double Power(double base, int exponent) {
        boolean isNegative = false;
        if(exponent < 0){
            isNegative = true;
            exponent = -exponent;
        }
        double result = 1;
        for(int i = 1; i <= exponent; i++){
            result = base * result;
        }
        return isNegative ? 1 / result : result;
    }
}
```

#### 递归
因为 (x*x)n/2 可以通过递归求解，并且每次递归 n 都减小一半，因此整个算法的时间复杂度为 O(logN)。
```java
public double Power(double base, int exponent) {
    if (exponent == 0)
        return 1;
    if (exponent == 1)
        return base;
    boolean isNegative = false;
    if (exponent < 0) {
        exponent = -exponent;
        isNegative = true;
    }
    double pow = Power(base * base, exponent / 2);
    if (exponent % 2 != 0)
        pow = pow * base;
    return isNegative ? 1 / pow : pow;
}
```

## 调整数组顺序使奇数位于偶数前面
### 题目描述
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。
### 题解
```java
public void reOrderArray(int[] nums) {
    int oddCount = 0;
    for (int val : nums)
        if (val % 2 == 1)
            oddCount++;
    int[] copy = nums.clone();
    int i = 0, j = oddCount;
    for (int num : copy) {
        if (num % 2 == 1)
            nums[i++] = num;
        else
            nums[j++] = num;
    }
}
```

## 链表中倒数第k个结点
### 题目描述
输入一个链表，输出该链表中倒数第k个结点。

### 题解

#### 遍历两遍
先遍历第一遍得出链表长度，然后得到倒数第k个为正数第n-k+1个，第二次遍历至该点输出
```java
public class Solution {
    public ListNode FindKthToTail(ListNode head,int k) {
        ListNode current = head;
        int i = 0;
        while (current != null){
            current = current.next;
            i++;
        }
        current = head;
        int rank = i - k;
        if (rank < 0)
            return null;
        for (int j = 0; j < rank; j++) {
            current = current.next;
        }
        return current;
    }
}
```

#### 遍历一遍
两个指针p1、p2，p1先动至正数第k个节点，此时p1，p2相差k个节点，p2开始动，当p1到表尾时，p2与p1相差k，即p2在倒数第k个结点
```java
public class Solution {
    public ListNode FindKthToTail(ListNode head,int k) {
        ListNode pre=null,p=null;
        //两个指针都指向头结点
        p=head;
        pre=head;
        //记录k值
        int a=k;
        //记录节点的个数
        int count=0;
        //p指针先跑，并且记录节点数，当p指针跑了k-1个节点后，pre指针开始跑，
        //当p指针跑到最后时，pre所指指针就是倒数第k个节点
        while(p!=null){
            p=p.next;
            count++;
            if(k<1){
                pre=pre.next;
            }
            k--;
        }
        //如果节点个数小于所求的倒数第k个节点，则返回空
        if(count<a) return null;
        return pre;
            
    }
}
```

## 反转链表

### 题目描述
输入一个链表，反转链表后，输出新链表的表头。

```java
public class Solution {
    public ListNode ReverseList(ListNode head) {
        if (head == null)
            return null;
        ListNode listNode = new ListNode(head.val);
        ListNode current = head.next;
        if (current == null)
            return listNode;
        ListNode resulthead = null;
        while (current != null){
            resulthead = new ListNode(current.val);
            resulthead.next = listNode;
            listNode = resulthead;
            current = current.next;
        }
        return resulthead;
    }
}
```

## 合并两个排序的链表
### 题目描述
输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

### 题解

#### 迭代
```java
public ListNode Merge(ListNode list1, ListNode list2) {
    ListNode head = new ListNode(-1);
    ListNode cur = head;
    while (list1 != null && list2 != null) {
        if (list1.val <= list2.val) {
            cur.next = list1;
            list1 = list1.next;
        } else {
            cur.next = list2;
            list2 = list2.next;
        }
        cur = cur.next;
    }
    if (list1 != null)
        cur.next = list1;
    if (list2 != null)
        cur.next = list2;
    return head.next;
}
```
#### 递归

```java
public ListNode Merge(ListNode list1, ListNode list2) {
    if (list1 == null)
        return list2;
    if (list2 == null)
        return list1;
    if (list1.val <= list2.val) {
        list1.next = Merge(list1.next, list2);
        return list1;
    } else {
        list2.next = Merge(list1, list2.next);
        return list2;
    }
}
```

## 树的子结构
### 题目描述
输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）此题要特别注意由于计算机表示小数含有误差，不能直接使用==进行double类型的等值判断，而是判断两个小数的差的绝对值是否小于某一个可忽略的数。
### 题解
```java
public class Solution {
    public boolean HasSubtree(TreeNode root1,TreeNode root2) {
        boolean result = false;
        
        if(root1 != null && root2 != null) {
        	if(Equal(root1.val, root2.val))
        		result = DoesTree1HaveTree2(root1, root2);
        	if(!result)
        		result = HasSubtree(root1.left, root2);
        	if(!result)
        		result = HasSubtree(root1.right, root2);
        }
        
        return result;
    }
    
    public boolean DoesTree1HaveTree2(TreeNode root1, TreeNode root2) {
    	if(root2 == null)
    		return true;
    	if(root1 == null)
    		return false;
    	
    	if(!Equal(root1.val, root2.val))
    		return false;
    	
    	return DoesTree1HaveTree2(root1.left, root2.left) && DoesTree1HaveTree2(root1.right, root2.right);
    }
    
    public boolean Equal(double num1, double num2) {
    	if(num1- num2 > -0.0000001 && num1 - num2 < 0.0000001)
    		return true;
    	return false;
    }
}
```

## 二叉树的镜像
### 题目描述
操作给定的二叉树，将其变换为源二叉树的镜像。
### 输入描述:
```
二叉树的镜像定义：源二叉树 

    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5
```
### 题解
```java
public class Solution {
    public void Mirror(TreeNode root) {
        if (root != null){
            TreeNode temp = root.left;
            root.left = root.right;
            root.right = temp;
            Mirror(root.left);
            Mirror(root.right);
        }
    }
}
```

## 顺时针打印矩阵
### 题目描述
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.
### 题解

```java
import java.util.ArrayList;
public class Solution {
    public ArrayList<Integer> printMatrix(int[][] matrix) {
    ArrayList<Integer> ret = new ArrayList<>();
    int r1 = 0, r2 = matrix.length - 1, c1 = 0, c2 = matrix[0].length - 1;
    while (r1 <= r2 && c1 <= c2) {
        for (int i = c1; i <= c2; i++)
            ret.add(matrix[r1][i]);
        for (int i = r1 + 1; i <= r2; i++)
            ret.add(matrix[i][c2]);
        if (r1 != r2)
            for (int i = c2 - 1; i >= c1; i--)
                ret.add(matrix[r2][i]);
        if (c1 != c2)
            for (int i = r2 - 1; i > r1; i--)
                ret.add(matrix[i][c1]);
        r1++; r2--; c1++; c2--;
    }
    return ret;
}
}
```

## 包含min函数的栈
### 题目描述
定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
### 题解
建立一个存储最小值的栈，栈顶为最小值，当弹出时判定两个栈的栈顶是否相同，如果相同则一起弹出，如果不同，只弹出数据栈。
```java
import java.util.Stack;

public class Solution {
    Stack<Integer> data = new Stack<>(); 
    Stack<Integer> min = new Stack<>();
    public void push(int node) {
        if (!min.empty()){
            if (node < min.peek())
                min.push(node);
        }
        else {
            min.push(node);
        }
        data.push(node);
    }

    public void pop() {
        if (data.peek() == min.peek()){
            min.pop();
        }
        data.pop();
    }

    public int top() {
        return data.peek();
    }

    public int min() {
        return min.peek();
    }
}
```

## 栈的压入、弹出序列
### 题目描述
输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）
### 题解
```java
import java.util.Stack;
public class Solution {
    public boolean IsPopOrder(int [] pushA,int [] popA) {
        if(pushA.length == 0 || popA.length == 0)
            return false;
        Stack<Integer> s = new Stack<Integer>();
        //用于标识弹出序列的位置
        int popIndex = 0;
        for(int i = 0; i< pushA.length;i++){
            s.push(pushA[i]);
            //如果栈不为空，且栈顶元素等于弹出序列
            while(!s.empty() &&s.peek() == popA[popIndex]){
                //出栈
                s.pop();
                //弹出序列向后一位
                popIndex++;
            }
        }
        return s.empty();
    }
}
```

## 从上往下打印二叉树
### 题目描述
从上往下打印出二叉树的每个节点，同层节点从左至右打印。
### 题解
使用队列结构，进行广度优先遍历
```java
import java.util.ArrayList;
/**
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/
import java.util.ArrayList;
import java.util.LinkedList;

public class Solution {
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
        LinkedList<TreeNode> queue = new LinkedList<>();
        ArrayList<Integer> result = new ArrayList<>();
        if (root == null)
            return result;
        queue.addLast(root);
        TreeNode temp;
        while (!queue.isEmpty()){
           temp = queue.removeFirst();
           result.add(temp.val);
           if (temp.left != null)
               queue.addLast(temp.left);
           if (temp.right != null)
               queue.addLast(temp.right);
        }
        return result;
    }
}
```

## 二叉搜索树的后序遍历序列
### 题目描述
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
### 题解
先以以下二叉树为例，其输入数组为{5, 7, 6, 9, 11, 10, 8}。
```
    8
   /  \
  6    10
 / \  / \
5  7 9   11

```
我们可以发现，数组的最后一个数字8就是二叉树的根节点，然后从数组开始进行遍历，凡是比8小的都属于根节点的左子树，其余的就是根节点的右子树，即{5, 7, 6, /9, 11, 10,/ 8}。我们在看看根节点的左子树，同样最后一个数字6是左子树的根节点，而5、7分别属于左子树根节点的左右子树。

再看看另一个例子：{7, 4, 6, 5}，由以上分析的规律可以发现，5为二叉树的根节点，而7、4、6都比5大，说明此二叉树没有左子树，而在右子树{7, 4, 6}中，7比6大，说明7在根节点的右子树中，而4却又比6小，这有违二叉树的定义，说明此数组不属于任何一个二叉树。

因此，我们可以使用递归来解决这个问题，先找到二叉树的根节点，再基于此根节点将数组拆分成左右子树，然后对左右子树分别进行递归。

```java
public class Solution {
    public boolean VerifySquenceOfBST(int [] sequence) {
        if(sequence.length==0)
            return false;
        if(sequence.length==1)
            return true;
        return ju(sequence, 0, sequence.length-1);
         
    }
     
    public boolean ju(int[] a,int star,int root){
        if(star>=root)
            return true;
        int i = root;
        //从后面开始找
        while(i>star&&a[i-1]>a[root])
            i--;//找到比根小的坐标
        //从前面开始找 star到i-1应该比根小
        for(int j = star;j<i-1;j++)
            if(a[j]>a[root])
                return false;;
        return ju(a,star,i-1)&&ju(a, i, root-1);
    }
}
```

## 二叉树中和为某一值的路径
### 题目描述
输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)
### 题解
此题运用深度优先搜索和回溯的思想。从左开始向下深度遍历，遇到叶节点之后，判断其值是否等于target，如果相等则将此路径加入到所有路径的列表中。每次回退的时候，都要将路径最后一个节点删除。
```java
import java.util.ArrayList;
/**
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/
public class Solution {
    private ArrayList<ArrayList<Integer>> listAll = new ArrayList<ArrayList<Integer>>();
    private ArrayList<Integer> list = new ArrayList<Integer>();
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
        if(root == null) return listAll;
        list.add(root.val);
        target -= root.val;
        if(target == 0 && root.left == null && root.right == null)
            listAll.add(new ArrayList<Integer>(list));
        FindPath(root.left, target);
        FindPath(root.right, target);
        list.remove(list.size()-1);
        return listAll;
    }
}
```

## 复杂链表的复制
### 题目描述
输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
### 题解
此题需要注意的是，不能单独新建链表一个个复制节点，因为无法确定复杂指针的位置。此题应该分三步：1、复制每个结点，如复制结点A得到A1，将结点A1插到结点A后面；2、重新遍历链表，复制老结点的随机指针给新结点，如A1.random = A.random.next;3、拆分链表，将链表拆分为原链表和复制后的链表
```java
/*
public class RandomListNode {
    int label;
    RandomListNode next = null;
    RandomListNode random = null;

    RandomListNode(int label) {
        this.label = label;
    }
}
*/
public class Solution {
   public RandomListNode Clone(RandomListNode pHead) {
        if(pHead == null) {
            return null;
        }
         
        RandomListNode currentNode = pHead;
        //1、复制每个结点，如复制结点A得到A1，将结点A1插到结点A后面；
        while(currentNode != null){
            RandomListNode cloneNode = new RandomListNode(currentNode.label);
            RandomListNode nextNode = currentNode.next;
            currentNode.next = cloneNode;
            cloneNode.next = nextNode;
            currentNode = nextNode;
        }
         
        currentNode = pHead;
        //2、重新遍历链表，复制老结点的随机指针给新结点，如A1.random = A.random.next;
        while(currentNode != null) {
            currentNode.next.random = currentNode.random==null?null:currentNode.random.next;
            currentNode = currentNode.next.next;
        }
         
        //3、拆分链表，将链表拆分为原链表和复制后的链表
        currentNode = pHead;
        RandomListNode pCloneHead = pHead.next;
        while(currentNode != null) {
            RandomListNode cloneNode = currentNode.next;
            currentNode.next = cloneNode.next;
            cloneNode.next = cloneNode.next==null?null:cloneNode.next.next;
            currentNode = currentNode.next;
        }
         
        return pCloneHead;
    }
}
```

## 二叉搜索树与双向链表
### 题目描述
输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
### 题解
```java
/**
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/
public class Solution {
   public TreeNode Convert(TreeNode root) {
        if(root == null)
            return null;
        if(root.left == null && root.right == null)
            return root;
        TreeNode left = Convert(root.left);
        TreeNode p = left;
        while(p != null && p.right != null) {
            p = p.right;
        }
        if(left != null) {
            root.left = p;
            p.right = root;
        }
        TreeNode right = Convert(root.right);
        if(right != null) {
            root.right = right;
            right.left = root;
        }
        return left != null ? left : root;
    }
}
```


## 字符串的排列
### 题目描述
输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
### 输入描述:
输入一个字符串,长度不超过9(可能有字符重复),字符只包括大小写字母。
### 题解
```java
import java.util.ArrayList;
import java.util.Collections;
public class Solution {
    public ArrayList<String> Permutation(String str) {
        ArrayList<String> list = new ArrayList<>();
        if(str == null || str.length() == 0)
            return list;
        Permutation(str.toCharArray(), 0, list);
        Collections.sort(list);
        return list;
    }

    public void Permutation(char[] c, int i, ArrayList<String> list) {
        if(i == c.length) {
            String str = String.valueOf(c);
            if(!list.contains(str))
                list.add(String.valueOf(c));
            return;
        } else {
            for(int j = i; j < c.length; j++) {
                swap(c, i, j);
                Permutation(c, i+1, list);
                swap(c, i, j);
            }
        }
    }

    public void swap(char[] c, int i, int j){
        char temp;
        temp = c[i];
        c[i] = c[j];
        c[j] = temp;
    }
}
```

## 数组中出现次数超过一半的数字
### 题目描述
数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
### 题解
#### 使用HashMap
```java
import java.util.HashMap;

public class Solution {
    public int MoreThanHalfNum_Solution(int [] array) {
        HashMap<Integer, Integer> data = new HashMap<>();
        for (int i :
                array) {
            if (data.keySet().contains(i)){
                int temp = data.get(i);
                data.put(i, temp+1);
            }
            else {
                data.put(i, 1);
            }
        }
        for (int i :
                data.keySet()) {
            if (data.get(i) > array.length / 2)
                return i;
        }
        return 0;
    }
}
```

#### 排序法
数组排序后，如果符合条件的数存在，则一定是数组中间那个数。（比如：1，2，2，2，3；或2，2，2，3，4；或2，3，4，4，4等等）
```java
import java.util.Arrays;
 
public class Solution {
    public int MoreThanHalfNum_Solution(int [] array) {
        Arrays.sort(array);
        int count=0;
         
        for(int i=0;i<array.length;i++){
            if(array[i]==array[array.length/2]){
                count++;
            }
        }
        if(count>array.length/2){
            return array[array.length/2];
        }else{
            return 0;
        }
         
    }
}
```

## 最小的K个数
### 题目描述
输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
### 题解
可以使用最大堆实现，每次与堆顶进行比较，如果小于堆顶，则弹出堆顶加入该数
```java
import java.util.ArrayList;
import java.util.Comparator;
import java.util.PriorityQueue;

public class Solution {
    public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
        ArrayList<Integer> result = new ArrayList<Integer>();
        int length = input.length;
        if(k > length || k == 0){
            return result;
        }
        PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(k, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return o2.compareTo(o1);
            }
        });
        for (int i = 0; i < length; i++) {
            if (maxHeap.size() != k) {
                maxHeap.offer(input[i]);
            } else if (maxHeap.peek() > input[i]) {
                Integer temp = maxHeap.poll();
                temp = null;
                maxHeap.offer(input[i]);
            }
        }
        for (Integer integer : maxHeap) {
            result.add(integer);
        }
        return result;
    }
}
```

## 连续子数组的最大和
### 题目描述
HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

### 题解
使用动态规划，定义一个数组a，a[i]为前i项最大子段和。如果a[i-1]<0,那么不管第i项多大都会给子段和有负负影响，此时可以抛弃前面的子段，反之a[i-1]>0，则会造成正影响，则a[i]=a[i-1]+array[i]

```java
public class Solution {
    public int FindGreatestSumOfSubArray(int[] array) {
        int [] f = new int[array.length];
        int result = Integer.MIN_VALUE;
        f[0] = array[0];
        for (int i = 1; i < array.length; i++) {
            if (f[i-1] > 0)
                f[i] = f[i-1]+array[i];
            else{
                f[i] = array[i];
            }
            if (f[i] > result)
                result = f[i];

        }
        return result;
    }
}
```
## 整数中1出现的次数（从1到n整数中1出现的次数）
### 题目描述
求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

```java
public class Solution {
    public int NumberOf1Between1AndN_Solution(int n) {
        if (n<=0) {
            return 0;
        }
        int sum=0;
        int higher=10;
        int lower=1;
        for (int i=0;i<9;i++) {
            // full loop of lower digit repeats
            int mul=n/higher;
            // each full loop of lower digit repeats exactly lower times
            sum+=mul*lower;
            // count non-full loop
            sum+=count(n%higher,lower,1);
            if (mul==0) {
                break;
            }
            if (i==8) {
                // count 10^9 digit
                sum+=count(n,higher,1);
            }
            higher*=10;
            lower*=10;
        }
        return sum;
    }

    int count(int val, int rate, int k) {
        // first digit
        int digit=val/rate;
        // first digit less than k, zero
        if (digit<k) {
            return 0;
        }
        // first digit greater than k, full loop
        if (digit>k) {
            return rate;
        }
        // first digit equals k
        return 1+(val%rate);
    }
}
```

## 把数组排成最小的数
### 题目描述
输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
### 题解
```java
public String PrintMinNumber(int [] numbers) {
        int n;
        String s="";
        ArrayList<Integer> list= new ArrayList<Integer>();
        n=numbers.length;
        for(int i=0;i<n;i++){
            list.add(numbers[i]);

        }
        Collections.sort(list, new Comparator<Integer>(){

            public int compare(Integer str1,Integer str2){
                String s1=str1+""+str2;
                String s2=str2+""+str1;
                return s1.compareTo(s2);
            }
        });

        for(int j:list){
            s+=j;
        }
        return s;

    }
```
## 丑数
### 题目描述
把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
### 题解
```java
public class Solution {
    public int GetUglyNumber_Solution(int n)
    {
        if(n<=0)return 0;
        ArrayList<Integer> list=new ArrayList<Integer>();
        list.add(1);
        int i2=0,i3=0,i5=0;
        while(list.size()<n)//循环的条件
        {
            int m2=list.get(i2)*2;
            int m3=list.get(i3)*3;
            int m5=list.get(i5)*5;
            int min=Math.min(m2,Math.min(m3,m5));
            list.add(min);
            if(min==m2)i2++;
            if(min==m3)i3++;
            if(min==m5)i5++;
        }
        return list.get(list.size()-1);
    }
}
```
    
## 第一个只出现一次的字符
### 题目描述
在一个字符串(0<=字符串长度<=10000，全部由字母组成)中找到第一个只出现一次的字符,并返回它的位置, 如果没有则返回 -1（需要区分大小写）.
### 题解
```java
public class Solution {
    public int FirstNotRepeatingChar(String str){
        char[] c = str.toCharArray();
        int[] a = new int['z'+1];
        for (char d : c)
            a[(int) d]++;
        for (int i = 0; i < c.length; i++)
            if (a[(int) c[i]] == 1)
                return i;
        return -1;
    }
}
```
## 数组中的逆序对
### 题目描述
在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007
输入描述:
题目保证输入的数组中没有的相同的数字
```
数据范围：

	对于%50的数据,size<=10^4

	对于%75的数据,size<=10^5

	对于%100的数据,size<=2*10^5

示例1
输入
1,2,3,4,5,6,7,0
输出
7
```

### 题解
```java
public class Solution {
    public int InversePairs(int [] array) {
        if(array==null || array.length<=0){
            return 0;
        }
        int pairsNum=mergeSort(array,0,array.length-1);
        return pairsNum;
    }

    public int mergeSort(int[] array,int left,int right){
        if(left==right){
            return 0;
        }
        int mid=(left+right)/2;
        int leftNum=mergeSort(array,left,mid);
        int rightNum=mergeSort(array,mid+1,right);
        return (Sort(array,left,mid,right)+leftNum+rightNum)%1000000007;
    }

    public int Sort(int[] array,int left,int middle,int right){
        int current1=middle;
        int current2=right;
        int current3=right-left;
        int temp[]=new int[right-left+1];
        int pairsnum=0;
        while(current1>=left && current2>=middle+1){
            if(array[current1]>array[current2]){
                temp[current3--]=array[current1--];
                pairsnum+=(current2-middle);     //这个地方是current2-middle！！！！
                if(pairsnum>1000000007)//数值过大求余
                {
                    pairsnum%=1000000007;
                }
            }else{
                temp[current3--]=array[current2--];
            }
        }
        while(current1>=left){
            temp[current3--]=array[current1--];
        }
        while(current2>=middle+1){
            temp[current3--]=array[current2--];
        }
        //将临时数组赋值给原数组
        int i=0;
        while(left<=right){
            array[left++]=temp[i++];
        }
        return pairsnum;
    }
}
```

## 两个链表的第一个公共结点
### 题目描述
输入两个链表，找出它们的第一个公共结点。

### 题解

两个链表存在公共节点，则可知链表呈Y字行，可以先计算两链表的长度差i，然后长链表指针前进i个结点，然后两个指针共同前进判断是否相等即可
```java
public ListNode FindFirstCommonNodeII(ListNode pHead1, ListNode pHead2) {
        ListNode current1 = pHead1;// 链表1
        ListNode current2 = pHead2;// 链表2
        if (pHead1 == null || pHead2 == null)
            return null;
        int length1 = getLength(current1);
        int length2 = getLength(current2);
        // 两连表的长度差

        // 如果链表1的长度大于链表2的长度
        if (length1 >= length2) {
            int len = length1 - length2;
            // 先遍历链表1，遍历的长度就是两链表的长度差
            while (len > 0) {
                current1 = current1.next;
                len--;
            }

        }
        // 如果链表2的长度大于链表1的长度
        else if (length1 < length2) {
            int len = length2 - length1;
            // 先遍历链表1，遍历的长度就是两链表的长度差
            while (len > 0) {
                current2 = current2.next;
                len--;
            }

        }
        //开始齐头并进，直到找到第一个公共结点
        while(current1!=current2){
            current1=current1.next;
            current2=current2.next;
        }
        return current1;

    }

    // 求指定链表的长度
    public static int getLength(ListNode pHead) {
        int length = 0;

        ListNode current = pHead;
        while (current != null) {
            length++;
            current = current.next;
        }
        return length;
    }
```

## 数字在排序数组中出现的次数
### 题目描述
统计一个数字在排序数组中出现的次数。
### 题解
先用二分查找找到k的位置，然后从i分别向前向后计算次数即可
```java
public class Solution {
    public int GetNumberOfK(int [] array , int k) {
        if (array == null || array.length == 0)
            return 0;
        int lo = 0, hi = array.length-1, mid;
        int index = 0, count = 0;
        while (lo <= hi){
            mid = lo + (hi - lo) / 2;
            if (k > array[mid]){
                lo = mid + 1;
            }
            else if (k < array[mid]){
                hi = mid - 1;
            }
            else {
                index = mid;
                break;
            }

        }
        for (int i = index; i >= 0 ; i--) {
            if (array[i] == k)
                count++;
            else
                break;
        }

        for (int i = index + 1; i < array.length; i++) {
            if (array[i] == k)
                count++;
            else
                break;
        }
        return count;
    }

}
```
## 二叉树的深度
### 题目描述
输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
### 题解
深度优先，访问子结点时传层数参数，到叶子结点与当前最大判断即可
```java
/**
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/
public class Solution {
    int max = 0;
    public int TreeDepth(TreeNode root) {
        if (root == null)
            return 0;
        BFS(root, 1);
        return max;
    }

    public void BFS(TreeNode root, int i){
        if (root.left == null && root.right == null){
            if (i > max)
                max = i;
        }
        if (root.left != null)
            BFS(root.left, i+1);
        if (root.right != null)
            BFS(root.right, i+1);
    }
}
```
## 平衡二叉树
### 题目描述
输入一棵二叉树，判断该二叉树是否是平衡二叉树。
### 题解
```java
public class Solution {
   boolean isBalanced = true;
    public boolean IsBalanced_Solution(TreeNode root) {
        search(root);
        return isBalanced;
    }

    public int search(TreeNode root){
        if (root == null)
            return 0;
        if (root.left == null && root.right == null){
            return 1;
        }
        int leftSum = search(root.left);
        int rightSum = search(root.right);
        if (leftSum - rightSum > 1 || rightSum - leftSum > 1)
            isBalanced = false;

        return Math.max(leftSum, rightSum) + 1;
    }
    
}
```
## 数组中只出现一次的数字
### 题目描述
一个整型数组里除了两个数字之外，其他的数字都出现了偶数次。请写程序找出这两个只出现一次的数字。
### 题解
(^)异或：当两个数相等则为0，有交换律。
由于数组其他数字出现偶数次，将他们分别做异或操作，根据交换律，最后结果只剩下两个数的异或值。
这个结果的二进制中的1，表现的是A和B的不同的位。我们就取第一个1所在的位数，假设是第3位，接着把原数组分成两组，分组标准是第3位是否为1。如此，相同的数肯定在一个组，因为相同数字所有位都相同，而不同的数，肯定不在一组。然后在其中一组的每个元素进行异或操作可以得出其中一个数，而另一个数用第一次的异或结果在与第一个数异或则可以得出
```java
public class Solution {
    public void FindNumsAppearOnce(int[] array, int[] num1, int[] num2)    {
        int sum = 0, k = 0;
        for (int i :
                array) {
            sum ^= i;
        }
        while ((sum >> k & 1) != 1) k++;
        int first = 0;
        for (int i :
                array) {
            if (((i >> k) & 1) == 1)
                first ^= i;
        }
        num1[0] = first;
        num2[0] = sum ^ first;
    }
}
```

## 和为S的连续正数序列
### 题目描述
小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!
输出描述:
### 题解
```java
import java.util.ArrayList;
public class Solution {
    public ArrayList<ArrayList<Integer> > FindContinuousSequence(int sum) {
        //存放结果
        ArrayList<ArrayList<Integer> > result = new ArrayList<>();
        //两个起点，相当于动态窗口的两边，根据其窗口内的值的和来确定窗口的位置和大小
        int plow = 1,phigh = 2;
        while(phigh > plow){
            //由于是连续的，差为1的一个序列，那么求和公式是(a0+an)*n/2
            int cur = (phigh + plow) * (phigh - plow + 1) / 2;
            //相等，那么就将窗口范围的所有数添加进结果集
            if(cur == sum){
                ArrayList<Integer> list = new ArrayList<>();
                for(int i=plow;i<=phigh;i++){
                    list.add(i);
                }
                result.add(list);
                plow++;
            //如果当前窗口内的值之和小于sum，那么右边窗口右移一下
            }else if(cur < sum){
                phigh++;
            }else{
            //如果当前窗口内的值之和大于sum，那么左边窗口右移一下
                plow++;
            }
        }
        return result;
    }
}
```

## 和为S的两个数字
### 题目描述
输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
### 题解
先使用二分查找找到该数或者比该数小的最大数的下标，然后使用双指针遍历
```java
import java.util.ArrayList;
public class Solution {
    public static ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {
        if (array == null || array.length == 0)
            return new ArrayList<Integer>();
        int lo = 0, hi = array.length - 1, mid = 0;
        while (hi >= lo){
            mid = lo + (hi - lo) / 2;
            if (sum > array[mid])
                lo = mid + 1;
            if (sum < array[mid])
                hi = mid - 1;
            if (sum == array[mid])
                break;
        }
        if (array[mid] > sum)
            mid--;
        int i = mid, j = 0;
        while (i != j){
            if (array[i] + array[j] == sum){
                ArrayList<Integer> result = new ArrayList<>();
                result.add(array[j]);
                result.add(array[i]);
                return result;
            }
            else if (array[i] + array[j] > sum){
                i--;
            }
            else if (array[i] + array[j] < sum){
                j++;
            }
        }
        return new ArrayList<Integer>();
    }
}

```

## 左旋转字符串
### 题目描述
汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！
### 题解
```
假设翻转的字符串为abcdef，n=3
设 X="abc" Y="def"
定义 X_为X翻转
则 X_="cba" Y_="fed"
X_Y_="cbafed" 而 (X_Y_)_="defabc" 三次翻转即可
```
```java
public class Solution {
    public String LeftRotateString(String str,int n) {
        char[] chars = str.toCharArray();
        if(chars.length < n) return "";
        reverse(chars, 0, n-1);
        reverse(chars, n, chars.length-1);
        reverse(chars, 0, chars.length-1);
        return new String(chars);
    }

    public void reverse(char[] chars,int low,int high){
        char temp;
        while(low<high){
            temp = chars[low];
            chars[low] = chars[high];
            chars[high] = temp;
            low++;
            high--;
        }
    }
}
```

## 翻转单词顺序列
### 题目描述
牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？
### 题解
先整体翻转，再对单词局部翻转
```java
public class Solution {
    public String ReverseSentence(String str) {
        if(str == null)
            return null;
        else if(str == "")
            return "";
        char[] data = str.toCharArray();
        int i = 0, j = data.length-1;
        reverse(data, i, j);
        j = 0;
        while(i < data.length) {
            if(j == data.length || data[j] == ' ') {
                reverse(data, i, j-1);
                i = j + 1;
            }
            j++;
        }
        return new String(data);
    }

    private void reverse(char[] data, int i, int j) {
        while(i <= j) {
            char temp = data[i];
            data[i] = data[j];
            data[j] = temp;
            i++;
            j--;
        }
    }
}
```

## 扑克牌顺子
### 题目描述
LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。为了方便起见,你可以认为大小王是0。
### 题解
```java
import java.util.Arrays;
public class Solution {
    public static boolean isContinuous(int [] numbers) {
        if (numbers == null)
            return false;
        if (numbers.length == 0)
            return false;
        Arrays.sort(numbers);
        int zerocount = 0;

        for (int i :
                numbers) {
            if (i == 0)
                zerocount++;
        }
        for (int i = zerocount; i < numbers.length-1; i++) {
            if (numbers[i] == numbers[i+1]) return false;
            if (numbers[i] != numbers[i+1]-1){
                if (zerocount >= numbers[i+1]-numbers[i]-1)
                    zerocount -= (numbers[i+1]-numbers[i]-1);
                else
                    return false;
            }
        }
        return true;
    }
}
```

## 孩子们的游戏(圆圈中最后剩下的数)
### 题目描述
每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)
### 题解
```java
import java.util.LinkedList;
public class Solution {
    public int LastRemaining_Solution(int n, int m) {
        LinkedList<Integer> list = new LinkedList<Integer>();
        for (int i = 0; i < n; i ++) {
            list.add(i);
        }

        int bt = 0;
        while (list.size() > 1) {
            bt = (bt + m - 1) % list.size();
            list.remove(bt);
        }

        return list.size() == 1 ? list.get(0) : -1;
    }
}
```
## 求1+2+3+...+n
### 题目描述
求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。
### 题解
```java
public class Solution {
        public static int Sum_Solution(int n) {
        int sum = n;
        boolean ans = (n>0)&&((sum+=Sum_Solution(n-1))>0);
        return sum;
    }
}
```

## 不用加减乘除做加法

### 题目描述
写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
### 题解
```java
public class Solution {
    public int Add(int num1,int num2) {
        while (num2!=0) {
            int temp = num1^num2;
            num2 = (num1&num2)<<1;
            num1 = temp;
        }
        return num1;
    }
}
```

## 把字符串转换成整数

### 题目描述
将一个字符串转换成一个整数(实现Integer.valueOf(string)的功能，但是string不符合数字要求时返回0)，要求不能使用字符串转换整数的库函数。 数值为0或者字符串不是一个合法的数值则返回0。

### 题解
```java
public class Solution {
    public static int StrToInt(String str) {
        if (str == null || str.trim().equals(""))
            return 0;
        int symbol = 1;
        if (str.charAt(0) == '-'){
            symbol = -1;
            str = str.substring(1);
        }
        else if (str.charAt(0) == '+')
            str = str.substring(1);
        int num = 1, len = str.length();
        long result = 0;
        int temp;
        while (num <= str.length()){
            temp = str.charAt(len - num);
            if (temp > '9' || temp < '0')
                return 0;
            temp -= 48;
            result += temp * Math.pow(10, num-1);
            num++;
        }
        return symbol * (int)result;
    }
}
```

## 数组中重复的数字
### 题目描述
在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

### 题解
```java
import java.util.Arrays;
public class Solution {
    
    // Parameters:
    //    numbers:     an array of integers
    //    length:      the length of array numbers
    //    duplication: (Output) the duplicated number in the array number,length of duplication array is 1,so using duplication[0] = ? in implementation;
    //                  Here duplication like pointor in C/C++, duplication[0] equal *duplication in C/C++
    //    这里要特别注意~返回任意重复的一个，赋值duplication[0]
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
     public boolean duplicate(int numbers[],int length,int [] duplication) {
        if (numbers == null || numbers.length == 0){
            duplication[0] = -1;
            return false;
        }
            
        Arrays.sort(numbers);
        for (int i = 0; i < length - 1; i++) {
            if (numbers[i] == numbers[i+1]){
                duplication[0] = numbers[i];
                return true;
            }
        }
        return false;
    }
}
```

## 构建乘积数组
### 题目描述
给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。
### 题解
```java
public class Solution {
    public int[] multiply(int[] A) {
        int length = A.length;
        int[] B = new int[length];
        if(length != 0 ){
            B[0] = 1;
            //计算下三角连乘
            for(int i = 1; i < length; i++){
                B[i] = B[i-1] * A[i-1];
            }
            int temp = 1;
            //计算上三角
            for(int j = length-2; j >= 0; j--){
                temp *= A[j+1];
                B[j] *= temp;
            }
        }
        return B;
    }
}
```

## 正则表达式匹配
### 题目描述
请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配
### 题解
```java
public class Solution {
    public boolean match(char[] str, char[] pattern) {
        boolean[][] dp = new boolean[str.length + 1][pattern.length + 1];
        dp[0][0] = true;
        for (int i = 1; i < dp[0].length; i ++) {
            if(pattern[i - 1] == '*') dp[0][i] = dp[0][i - 2];
        }
        for (int i = 1; i < dp.length; i ++) {
            for (int j = 1; j < dp[0].length; j ++) {
                if(pattern[j - 1] == '.' || pattern[j - 1] == str[i - 1]) dp[i][j] = dp[i - 1][j - 1];
                else if(pattern[j - 1] == '*') {
                    if(pattern[j - 2] != str[i - 1] && pattern[j - 2] != '.') dp[i][j] = dp[i][j - 2];
                    else dp[i][j] = dp[i][j - 1] || dp[i][j - 2] || dp[i - 1][j];
                }
            }
        }
        return dp[str.length][pattern.length];
    }
}
```

## 表示数值的字符串
### 题目描述
请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。
### 题解
```java
public class Solution {
    private int index = 0;

    public boolean isNumeric(char[] str) {
        if (str.length < 1)
            return false;

        boolean flag = scanInteger(str);

        if (index < str.length && str[index] == '.') {
            index++;
            flag = scanUnsignedInteger(str) || flag;
        }

        if (index < str.length && (str[index] == 'E' || str[index] == 'e')) {
            index++;
            flag = flag && scanInteger(str);
        }

        return flag && index == str.length;

    }

    private boolean scanInteger(char[] str) {
        if (index < str.length && (str[index] == '+' || str[index] == '-') )
            index++;
        return scanUnsignedInteger(str);

    }

    private boolean scanUnsignedInteger(char[] str) {
        int start = index;
        while (index < str.length && str[index] >= '0' && str[index] <= '9')
            index++;
        return start < index; //是否存在整数
    }
}

```

## 字符流中第一个不重复的字符
### 题目描述
请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。
输出描述:
如果当前字符流没有存在出现一次的字符，返回#字符。
### 题解
```java
public class Solution {
    int[] table = new int[256];
    StringBuffer s=new StringBuffer();
    //Insert one char from stringstream
    public void Insert(char ch)
    {
        s.append(ch);
        if(table[ch] == 0)
            table[ch] = 1;
        else 
            table[ch] += 1;
    }
    //return the first appearence once char in current stringstream
    public char FirstAppearingOnce()
    {
        char[] str=s.toString().toCharArray();
        for(char c:str)
        {
            if(table[c]==1)
                return c;
        }
        return '#';
    }
}
```
## 链表中环的入口结点
### 题目描述
给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null。
### 题解
```java
/*
 public class ListNode {
    int val;
    ListNode next = null;

    ListNode(int val) {
        this.val = val;
    }
}
*/
public class Solution {
    ListNode EntryNodeOfLoop(ListNode pHead){
        if(pHead == null || pHead.next == null)
            return null;
        ListNode p1 = pHead;
        ListNode p2 = pHead;
        while(p2 != null && p2.next != null ){
            p1 = p1.next;
            p2 = p2.next.next;
            if(p1 == p2){
                p2 = pHead;
                while(p1 != p2){
                    p1 = p1.next;
                    p2 = p2.next;
                }
                if(p1 == p2)
                    return p1;
            }
        }
        return null;
    }
}
```

## 删除链表中重复的结点
### 题目描述
在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5
### 题解
```java
/*
 public class ListNode {
    int val;
    ListNode next = null;

    ListNode(int val) {
        this.val = val;
    }
}
*/
public class Solution {
    public ListNode deleteDuplication(ListNode pHead) { 
        if (pHead==null || pHead.next==null){
            return pHead;
        }
        ListNode Head = new ListNode(0);
        Head.next = pHead;
        ListNode pre  = Head;
        ListNode last = Head.next;
        while (last!=null){
            if(last.next!=null && last.val == last.next.val){
                // 找到最后的一个相同节点
                while (last.next!=null && last.val == last.next.val){
                    last = last.next;
                }
                pre.next = last.next;
                last = last.next;
            }else{
                pre = pre.next;
                last = last.next;
            }
        }
        return Head.next;
    }
}
```

## 二叉树的下一个结点
### 题目描述
给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
### 题解
```java
/*
public class TreeLinkNode {
    int val;
    TreeLinkNode left = null;
    TreeLinkNode right = null;
    TreeLinkNode next = null;

    TreeLinkNode(int val) {
        this.val = val;
    }
}
*/
import java.util.Stack;
public class Solution {
    public TreeLinkNode GetNext(TreeLinkNode pNode)
    {
        //1、一个节点有右子树，那么找到右子树的最左子节点
        if (pNode.right != null) {
            TreeLinkNode node = pNode.right;
            while (node.left != null) {
                node = node.left;
            }
            return node;
        }

        //2、一个节点没有右子树
        while (pNode.next != null) {
            if(pNode.next.left == pNode) return pNode.next;
            pNode = pNode.next;
        }
        return null;
    }
}
```

## 对称的二叉树
### 题目描述
请实现一个函数，用来判断一颗二叉树是不是对称的。注意，如果一个二叉树同此二叉树的镜像是同样的，定义其为对称的。
### 题解
```java
/*
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/

public class Solution {
    boolean isSymmetrical(TreeNode pRoot)
    {
        if(pRoot == null) return true;
        return isSymmetrical(pRoot.left, pRoot.right);
    }

    private boolean isSymmetrical(TreeNode left, TreeNode right) {
        if(left == null && right == null) return true;
        if(left == null || right == null) return false;
        return left.val == right.val //为镜像的条件：左右节点值相等
                && isSymmetrical(left.left, right.right) //2.对称的子树也是镜像
                && isSymmetrical(left.right, right.left);
    }
}
```

## 按之字形顺序打印二叉树
### 题目描述
请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。
### 题解
```java
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;

public class Solution {
    public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
        if (pRoot == null) {
            return ret;
        }
        ArrayList<Integer> list = new ArrayList<>();
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.addLast(null);//层分隔符
        queue.addLast(pRoot);
        boolean leftToRight = true;

        while (queue.size() != 1) {
            TreeNode node = queue.removeFirst();
            if (node == null) {//到达层分隔符
                Iterator<TreeNode> iter = null;
                if (leftToRight) {
                    iter = queue.iterator();//从前往后遍历
                } else {
                    iter = queue.descendingIterator();//从后往前遍历
                }
                leftToRight = !leftToRight;
                while (iter.hasNext()) {
                    TreeNode temp = (TreeNode)iter.next();
                    list.add(temp.val);
                }
                ret.add(new ArrayList<Integer>(list));
                list.clear();
                queue.addLast(null);//添加层分隔符
                continue;//一定要continue
            }
            if (node.left != null) {
                queue.addLast(node.left);
            }
            if (node.right != null) {
                queue.addLast(node.right);
            }
        }

        return ret;
    }
}

```

## 把二叉树打印成多行
### 题目描述
从上到下按层打印二叉树，同一层结点从左至右输出。每一层输出一行。
### 题解
```java
import java.util.ArrayList;


/*
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/
import java.util.Iterator;
import java.util.LinkedList;

public class Solution {
    public ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
        ArrayList<ArrayList<Integer>> ret = new ArrayList<>();
        if (pRoot == null) {
            return ret;
        }
        ArrayList<Integer> list = new ArrayList<>();
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.addLast(null);//层分隔符
        queue.addLast(pRoot);
        boolean leftToRight = true;

        while (queue.size() != 1) {
            TreeNode node = queue.removeFirst();
            if (node == null) {//到达层分隔符
                Iterator<TreeNode> iter = null;
                if (leftToRight) {
                    iter = queue.iterator();//从前往后遍历
                } else {
                    iter = queue.descendingIterator();//从后往前遍历
                }
                while (iter.hasNext()) {
                    TreeNode temp = (TreeNode)iter.next();
                    list.add(temp.val);
                }
                ret.add(new ArrayList<Integer>(list));
                list.clear();
                queue.addLast(null);//添加层分隔符
                continue;//一定要continue
            }
            if (node.left != null) {
                queue.addLast(node.left);
            }
            if (node.right != null) {
                queue.addLast(node.right);
            }
        }
        return ret;
    }
}

```

## 序列化二叉树
### 题目描述
请实现两个函数，分别用来序列化和反序列化二叉树

### 题解
```java
/*
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/
public class Solution {
     
    String Serialize(TreeNode root) {
        if(root == null)
            return "";
        StringBuilder sb = new StringBuilder();
        Serialize2(root, sb);
        return sb.toString();
    }
     
    void Serialize2(TreeNode root, StringBuilder sb) {
        if(root == null) {
            sb.append("#,");
            return;
        }
        sb.append(root.val);
        sb.append(',');
        Serialize2(root.left, sb);
        Serialize2(root.right, sb);
    }
     
    int index = -1;
     
    TreeNode Deserialize(String str) {
        if(str.length() == 0)
            return null;
        String[] strs = str.split(",");
        return Deserialize2(strs);
    }  
     
    TreeNode Deserialize2(String[] strs) {
        index++;
        if(!strs[index].equals("#")) {
            TreeNode root = new TreeNode(0);     
            root.val = Integer.parseInt(strs[index]);
            root.left = Deserialize2(strs);
            root.right = Deserialize2(strs);
            return root;
        }
        return null;
    }
     
}
```
## 二叉搜索树的第k个结点
### 题目描述
给定一棵二叉搜索树，请找出其中的第k小的结点。例如， （5，3，7，2，4，6，8）    中，按结点数值大小顺序第三小结点的值为4。
### 题解
```java
/*
public class TreeNode {
    int val = 0;
    TreeNode left = null;
    TreeNode right = null;

    public TreeNode(int val) {
        this.val = val;

    }

}
*/
import java.util.Stack;
public class Solution {
    TreeNode KthNode(TreeNode pRoot, int k) {
        int temp = 0;
        Stack<TreeNode> stack = new Stack<>();
        while (!stack.isEmpty() || pRoot != null){
            while (pRoot != null){
                stack.push(pRoot);
                pRoot = pRoot.left;
            }
            if (!stack.empty()){
                pRoot = stack.pop();
                if (++temp == k){
                    return pRoot;
                }
                pRoot = pRoot.right;
            }
        }
        return null;
    }
}
```

## 数据流中的中位数
### 题目描述
如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。我们使用Insert()方法读取数据流，使用GetMedian()方法获取当前读取数据的中位数。
### 题解
```java

import java.util.ArrayList;
import java.util.Collections;

public class Solution {

    private ArrayList<Integer> arrayList = new ArrayList<>();

    public void Insert(Integer num) {
        arrayList.add(num);
    }

    public Double GetMedian() {
        Collections.sort(arrayList);
        if (arrayList.size() % 2 == 1)
            return (double)arrayList.get(arrayList.size() / 2);
        else
            return (double)(arrayList.get(arrayList.size()/2) + arrayList.get(arrayList.size()/2-1))/2;
    }


}
```
## 滑动窗口的最大值
### 题目描述
给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。
### 题解
```java
import java.util.ArrayList;
import java.util.Comparator;
import java.util.PriorityQueue;

public class Solution {
    public static ArrayList<Integer> maxInWindows(int [] num, int size) {
        if (size > num.length || size == 0)
            return new ArrayList<Integer>();
        ArrayList<Integer> result = new ArrayList<>();
        PriorityQueue<Integer> queue = new PriorityQueue<>(size, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return o2-o1;
            }
        });
        for (int i = 0; i < size; i++) {
            queue.offer(num[i]);
        }
        result.add(queue.peek());
        int lru = 0;
        while (size < num.length){
            queue.remove(num[lru++]);
            queue.offer(num[size]);
            result.add(queue.peek());
            size++;
        }
        return result;
    }
   
}
```

## 矩阵中的路径
### 题目描述
请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。 例如 a b c e s f c s a d e e 这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
### 题解
```java
public class Solution {
    public boolean hasPath(char[] matrix, int rows, int cols, char[] str)
    {
        //标志位，初始化为false
        boolean[] flag = new boolean[matrix.length];
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                //循环遍历二维数组，找到起点等于str第一个元素的值，再递归判断四周是否有符合条件的----回溯法
                if(judge(matrix,i,j,rows,cols,flag,str,0)){
                    return true;
                }
            }
        }
        return false;
    }

    //judge(初始矩阵，索引行坐标i，索引纵坐标j，矩阵行数，矩阵列数，待判断的字符串，字符串索引初始为0即先判断字符串的第一位)
    private boolean judge(char[] matrix,int i,int j,int rows,int cols,boolean[] flag,char[] str,int k){
        //先根据i和j计算匹配的第一个元素转为一维数组的位置
        int index = i*cols+j;
        //递归终止条件
        if(i<0 || j<0 || i>=rows || j>=cols || matrix[index] != str[k] || flag[index] == true)
            return false;
        //若k已经到达str末尾了，说明之前的都已经匹配成功了，直接返回true即可
        if(k == str.length-1)
            return true;
        //要走的第一个位置置为true，表示已经走过了
        flag[index] = true;

        //回溯，递归寻找，每次找到了就给k加一，找不到，还原
        if(judge(matrix,i-1,j,rows,cols,flag,str,k+1) ||
                judge(matrix,i+1,j,rows,cols,flag,str,k+1) ||
                judge(matrix,i,j-1,rows,cols,flag,str,k+1) ||
                judge(matrix,i,j+1,rows,cols,flag,str,k+1)  )
        {
            return true;
        }
        //走到这，说明这一条路不通，还原，再试其他的路径
        flag[index] = false;
        return false;
    }


}
```

## 机器人的运动范围
### 题目描述
地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？
### 题解
```java
public class Solution {
    int count = 0;
    public int movingCount(int threshold, int rows, int cols)
    {
        boolean[][] flag = new boolean[rows][cols];
        getCount(flag, rows, cols, 0, 0, threshold);
        return count;
    }

    public void getCount(boolean[][] flag, int rows, int cols, int x, int y, int threshold){
        if (x < 0 || y < 0 || x >= rows || y >= cols || flag[x][y] == true){
            return;
        }
        flag[x][y] = true;
        if (sum(x, y) <= threshold){
            count++;
        }
        else {
            return;
        }


        getCount(flag, rows, cols, x+1, y, threshold);
        getCount(flag, rows, cols, x-1, y, threshold);
        getCount(flag, rows, cols, x, y+1, threshold);
        getCount(flag, rows, cols, x, y-1, threshold);
    }

    public static int sum(int x, int y){
        int sum = 0;
        while (x > 0){
            sum += (x % 10);
            x /= 10;
        }
        while (y > 0){
            sum += (y % 10);
            y /= 10;
        }
        return sum;
    }

  
}
``` 
