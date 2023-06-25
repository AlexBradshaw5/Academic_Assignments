import org.w3c.dom.ls.LSException;

import java.awt.geom.Arc2D;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Assignment1 {

    public void selectionSort(double[] arr){
        int min;
        for (int i = 0; i < arr.length; i++) {
            min = i;
            for (int j = i; j < arr.length; j++) {
                if(arr[j] < arr[min]){
                    min = j;
                }
            }
            swap(arr,i,min);
        }
    }

    public void bubbleSort(double[] arr){
        int i = 1;
        int e = arr.length;
        boolean flag = false;
        while(flag == false){
            flag = true;
            for (int j = i; j < e; j++) {
                if(arr[j-1] > arr[j]){
                    flag = false;
                    swap(arr,j-1,j);
                }
            }
            e--;
        }
    }

    public void insertionSort(double[] arr){
        for (int i = 1; i < arr.length; i++) {
            double I = arr[i];
            for (int j = i-1; j >= 0 ; j--) {
                if(I < arr[j]){
                    arr[j+1] = arr[j];
                    if(j == 0){
                        arr[j] = I;
                    }
                }
                else{
                    arr[j+1] = I;
                    j = -1;
                }
            }
        }
    }

    public void swap(double[] arr, int bot, int top){
        double temp = arr[bot];
        arr[bot] = arr[top];
        arr[top] = temp;
    }

    public int findPivot(double[] arr, int bot,int top){
        double a = arr[bot]; //1
        double b = arr[top];//2
        double c = arr[(bot+top)/2];//1
        if(a >= b && a >= c){//a is the greatest
            if(b<c){
                return (bot+top)/2;
            }
            else{
                return top;
            }
        }
        else if(b >= a && b >= c){//b is the greatest
            if(a<c){
                return (bot+top)/2;
            }
            else{
                return bot;
            }
        }
        else{ //c is greatest
            if(a<b){
                return top;
            }
            else{
                return bot;
            }
        }
    }

    public int partition(double[] arr, int bot, int top){
        if(bot<top) {
            int pivotI = findPivot(arr, bot, top); //pivot index
            swap(arr,pivotI,top);
            pivotI = top;
            int botIndex = bot;
            int topIndex = top - 1;
            while (botIndex <= topIndex) {
                while (botIndex < top && arr[botIndex] < arr[pivotI]) {
                    botIndex++;
                }
                while (topIndex >= botIndex && arr[topIndex] >= arr[pivotI]) {
                    topIndex--;
                }
                if (botIndex < topIndex) {
                    swap(arr, botIndex, topIndex);
                }
                else {
                    swap(arr, botIndex, pivotI);
                }
            }
            return botIndex;
        }
        return bot;
    }


    public void quicksort(double[] arr,int bot,int top){
        if(bot<top){
            int p = partition(arr,bot,top);
            quicksort(arr,bot,p-1);
            quicksort(arr,p+1,top);
        }
    }

    public double[] getLHS(double[] arr){
        int len = arr.length/2;
        double[] LHS = new double[len];
        for (int i = 0; i < len; i++) {
            LHS[i] = arr[i];
        }
        return LHS;
    }

    public double[] getRHS(double[] arr){
        int len = arr.length - arr.length/2;
        double[] RHS = new double[len];
        for (int i = 0; i < len; i++) {
            RHS[len-i-1] = arr[arr.length-i-1];
        }
        return RHS;
    }

    public void merge(double[] arr,double[] LHS,double[] RHS){
        int AI = 0;
        int LI = 0;
        int RI = 0;
        while(AI<arr.length){
            if(LHS[LI] <= RHS[RI]){
                arr[AI++] = LHS[LI++];
                if(LI==LHS.length){
                    while(RI < RHS.length) {
                        arr[AI++] = RHS[RI++];
                    }
                }
            }
            else{
                arr[AI++] = RHS[RI++];
                if (RI == RHS.length){
                    while(LI < LHS.length) {
                        arr[AI++] = LHS[LI++];
                    }
                }
            }
        }
    }

    public void mergeSort(double[] arr){
        if(arr.length > 1){
            double[] LHS = getLHS(arr);
            double[] RHS = getRHS(arr);
            mergeSort(LHS);
            mergeSort(RHS);
            merge(arr,LHS,RHS);
        }
    }

    public List<Double> quickerThan(double[] arr){
        List<List<Double>> sorted = findSorted(arr);
        while(sorted.get(0).size() < arr.length){//making first element the array we want
            List<List<Double>> temp = new ArrayList<>();
            int i = 0;//starting index
            int makeEven = sorted.size()/2*2;//to account for odd amount of elements
            while(i < makeEven){
                temp.add(newMerge(sorted.get(i), sorted.get(i+1)));//this groups together every pair
                i+=2;
            }
            if(i+1 == sorted.size()){//size was odd so one pair left out
                temp.add(sorted.get(i));//add to list
            }
            sorted = temp;//points to new list of sorted regions
        }
        return sorted.get(0);
    }

    public List<Double> newMerge(List<Double> one, List<Double> two){
        int oneIndex = 0;
        int twoIndex = 0;
        List<Double> temp = new ArrayList<>();
        while(temp.size() < one.size() + two.size()){
            if(one.get(oneIndex) <= two.get(twoIndex)){//if
                temp.add(one.get(oneIndex++));
                while(oneIndex >= one.size() && twoIndex < two.size()){
                    temp.add(two.get(twoIndex++));
                }
            }
            else{
                temp.add(two.get(twoIndex++));
                while(twoIndex >= two.size() && oneIndex < one.size()){
                    temp.add(one.get(oneIndex++));
                }
            }
        }
        return temp;
    }

    public List<List<Double>> findSorted(double[] arr){
        List<List<Double>> sorted = new ArrayList<>();
        int index = 0;
        boolean Last = false;
        List<Double> last = new ArrayList<>();
        while(index < arr.length-1){
            List<Double> temp = new ArrayList<>();
            temp.add(arr[index]);
            while(arr[index] <= arr[index+1] && index<arr.length-2){//stops when index+1 has no value
                temp.add(arr[++index]);
            }
            if(index == arr.length-2){//index is at second to last element
                if(arr[index] <= arr[index+1]){//final element is still in sorted portion
                    temp.add(arr[++index]);//add last element

                    //index++;//increment to final element
                }
                else{//final element is it's own sorted portion
                    Last = true;//signal if statement
                    last.add(arr[++index]);
                    //increment to final element
                }
            }
            index++;//end of sorted portion
            sorted.add(temp);
            if(Last == true){
                sorted.add(last);
            }
        }
        return sorted;
    }

    public boolean isSorted(double [] arr){
        for (int i = 0; i < arr.length-1; i++) {
            if(arr[i] > arr[i+1]){
                System.out.println("Array not sorted");
                return false;
            }
        }
        return true;
    }

    public boolean isSorted(List<Double> arr){
        for (int i = 0; i < arr.size()-1; i++) {
            if(arr.get(i) > arr.get(i+1)){
                System.out.println("Array not sorted");
                return false;
            }
        }
        return true;
    }

    public static double[] makeList(double[] arr){
        Random random = new Random();
        for (int i = 0; i < arr.length; i++) {
            arr[i] = random.nextDouble()*1000;
        }
        return arr;
    }

    public static void main(String[] args) {
        Assignment1 a1 = new Assignment1();
//        double[] lol = new double[5];
//        makeList(lol);
//        System.out.println(a1.findSorted(lol));
//        List<Double> newList = a1.quickerThan(lol);
//        System.out.println(newList);
//        for (int i = 0; i < lol.length; i++) {
//            System.out.print("[" + lol[i] + "], ");
//        }

        int l = 50000;
        double[] pt2;
        for (int y = 1; y < 11; y++) {
            pt2 = new double[y*l];
            makeList(pt2);
            long mystart = System.currentTimeMillis();
            List<Double> test = a1.quickerThan(pt2);
            System.out.println(System.currentTimeMillis() - mystart + " with elements: " + y*l);
            if (!a1.isSorted(test)){
                return;
            }
        }
        boolean isayso = true;
        if(isayso == true){
            return;
        }


        //part 1 below


        long insertTime;
        long bubbleTime;
        long mergeTime;
        long selectionTime;
        long quickTime;
        double[] List;
        double[] copy;
        for (int x = 1; x < 11; x++) {
            List = new double[x*l];
            makeList(List);

            copy = List.clone();
            long starti = System.currentTimeMillis();
            a1.insertionSort(copy);
            if (!a1.isSorted(copy)){
                return;
            }
            insertTime = System.currentTimeMillis() - starti;
            System.out.println(insertTime + " Insertion Sort with elements: " + x*l);

            copy = List.clone();
            long startb = System.currentTimeMillis();
            a1.bubbleSort(copy);
            bubbleTime = System.currentTimeMillis()- startb;
            if (!a1.isSorted(copy)){
                return;
            }
            System.out.println(bubbleTime + " Bubble Sort with elements: " + x*l);

            copy = List.clone();
            long startm = System.currentTimeMillis();
            a1.mergeSort(copy);
            mergeTime = System.currentTimeMillis() - startm;
            if (!a1.isSorted(copy)){
                System.out.println("Merge Sort");
                return;
            }
            System.out.println(mergeTime + " Merge Sort with elements: " + x*l);

            copy = List.clone();
            long starts = System.currentTimeMillis();
            a1.selectionSort(copy);
            selectionTime = System.currentTimeMillis() - starts;
            if (!a1.isSorted(copy)){
                return;
            }
            System.out.println(selectionTime + " Selection Sort with elements: " + x*l);

            copy = List.clone();
            long startq = System.currentTimeMillis();
            a1.quicksort(copy,0,List.length-1);
            quickTime = System.currentTimeMillis() - startq;
            if (!a1.isSorted(copy)){
                return;
            }
            System.out.println(quickTime + " Quick Sort with elements: " + x*l);
        }
    }
}
