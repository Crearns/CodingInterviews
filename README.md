# 《剑指offer》题解
更新时间：2019/2/22
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

