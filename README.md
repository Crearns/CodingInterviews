# 《剑指offer》题解
更新时间：2019/1/31
# content
* [二维数组中的查找](#二维数组中的查找)
* [从尾到头打印链表](#从尾到头打印链表)
* [重建二叉树](#重建二叉树)



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
