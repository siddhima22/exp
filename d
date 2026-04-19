//RAYMOND TREE
import java.util.*;

class Raymond {
    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        int[] parent = { -1, -1, 1, 1, 2, 2, 3, 3 };
        int tokenHolder = 1;

        System.out.print("Enter requesting process: ");
        int r = sc.nextInt();

        // -------- CALCULATE PATH ONCE --------
        int[] path = new int[10];
        int i = 0, t = r;

        while (t != -1) {
            path[i++] = t;
            t = parent[t];
        }

        // -------- REQUEST (UPWARD) --------
        for (int j = 0; j < i - 1; j++) {
            System.out.println("P" + path[j] + " sends REQUEST to P" + path[j + 1]);
            // System.out.println("Request Queue at P" + path[j + 1] + ": [P" + r + "]");
        }

        // -------- TOKEN (DOWNWARD) --------
        System.out.println("\nToken at P" + tokenHolder);
        for (int j = i - 1; j > 0; j--) {
            System.out.println("P" + path[j] + " sends TOKEN to P" + path[j - 1]);
            // System.out.println("Request Queue at P" + path[j] + ": served P" + r);
        }
        tokenHolder = r;
        // -------- CS --------
        System.out.println("\nP" + r + " enters CS");
        // System.out.println("Request Queue at P" + r + ": []");
        System.out.println("P" + r + " exits CS");
    }
}



//BERKLEY
import java.util.*;

class Berkeley {
    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.println("Enter Master time (h m s):");
        int mh = sc.nextInt(), mm = sc.nextInt(), ms = sc.nextInt();

        System.out.println("Enter number of nodes:");
        int n = sc.nextInt();

        int h[] = new int[n], m[] = new int[n], s[] = new int[n];
        double diff[] = new double[n];

        // Convert master time to seconds
        int masterTime = mh * 3600 + mm * 60 + ms;

        double sum = 0;

        System.out.println("Enter node times:");
        for (int i = 0; i < n; i++) {
            h[i] = sc.nextInt();
            m[i] = sc.nextInt();
            s[i] = sc.nextInt();

            int nodeTime = h[i] * 3600 + m[i] * 60 + s[i];

            diff[i] = nodeTime - masterTime; // difference
            sum += diff[i];
        }

        // Include master (diff = 0)
        double avg = sum / (n + 1);

        // Adjust master
        int newMaster = masterTime + (int) Math.round(avg);

        System.out.println("\nSynchronized Times:");
        System.out.println("Master: " + format(newMaster));

        // Adjust nodes
        for (int i = 0; i < n; i++) {
            int nodeTime = h[i] * 3600 + m[i] * 60 + s[i];

            int newTime = nodeTime - (int) Math.round(diff[i] - avg);

            System.out.println("Node " + (i + 1) + ": " + format(newTime));
        }

        sc.close();
    }

    // Convert seconds → HH:MM:SS
    static String format(int t) {
        t = (t + 86400) % 86400;
        int h = t / 3600;
        int m = (t % 3600) / 60;
        int s = t % 60;
        return String.format("%02d:%02d:%02d", h, m, s);
    }
}



//BANKER 
import java.util.*;

class Bankers {
    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Processes: ");
        int p = sc.nextInt();

        System.out.print("Resources: ");
        int r = sc.nextInt();

        int allocation[][] = new int[p][r];
        int max[][] = new int[p][r];
        int available[] = new int[r];
        int need[][] = new int[p][r];

        System.out.println("Enter Allocation:");
        for (int i = 0; i < p; i++)
            for (int j = 0; j < r; j++)
                allocation[i][j] = sc.nextInt();

        System.out.println("Enter Max:");
        for (int i = 0; i < p; i++)
            for (int j = 0; j < r; j++)
                max[i][j] = sc.nextInt();

        System.out.println("Enter Available:");
        for (int i = 0; i < r; i++)
            available[i] = sc.nextInt();

        // Need = Max - Allocation
        for (int i = 0; i < p; i++)
            for (int j = 0; j < r; j++)
                need[i][j] = max[i][j] - allocation[i][j];

        boolean finished[] = new boolean[p];
        int safeSeq[] = new int[p];
        int work[] = available.clone();

        int count = 0;

        while (count < p) {
            boolean found = false;

            for (int i = 0; i < p; i++) {
                if (!finished[i]) {

                    int j;
                    for (j = 0; j < r; j++)
                        if (need[i][j] > work[j])
                            break;

                    if (j == r) {
                        for (int k = 0; k < r; k++)
                            work[k] += allocation[i][k];

                        safeSeq[count++] = i;
                        finished[i] = true;
                        found = true;
                    }
                }
            }

            if (!found) {
                System.out.println("Not Safe");
                return;
            }
        }

        System.out.print("Safe Sequence: ");
        for (int i = 0; i < p; i++)
            System.out.print("P" + safeSeq[i] + " ");
    }
}



// GROUP CLIENT
import java.io.*;
import java.net.*;
import java.util.Scanner;

public class GroupClient {

    public static void main(String[] args) throws Exception {

        Socket socket = new Socket("localhost", 9999);

        Scanner in = new Scanner(socket.getInputStream());
        PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
        Scanner sc = new Scanner(System.in);

        System.out.print("Enter your name: ");
        String name = sc.nextLine();

        // Thread to receive messages
        new Thread(() -> {
            try {
                while (in.hasNextLine()) {
                    String msg = in.nextLine();
                    System.out.println(msg);
                }
            } catch (Exception e) {
            }
        }).start();

        // Send messages
        while (true) {
            String message = sc.nextLine();
            out.println(name + ": " + message);
        }
    }
}

//GOUP SERVER
import java.io.*;
import java.net.*;
import java.util.*;

public class GroupServer {

    static ArrayList<PrintWriter> clients = new ArrayList<>();

    public static void main(String[] args) throws Exception {

        ServerSocket serverSocket = new ServerSocket(9999);
        System.out.println("Server started...");

        while (true) {
            Socket socket = serverSocket.accept();
            System.out.println("New client connected");

            PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
            clients.add(out);

            new ClientHandler(socket).start();
        }
    }

    static class ClientHandler extends Thread {
        Scanner in;

        ClientHandler(Socket socket) throws Exception {
            in = new Scanner(socket.getInputStream());
        }

        public void run() {
            try {
                while (in.hasNextLine()) {
                    String msg = in.nextLine();

                    System.out.println("Message: " + msg);

                    for (PrintWriter pw : clients) {
                        pw.println(msg);
                    }
                }
            } catch (Exception e) {
                System.out.println("Client disconnected");
            }
        }
    }
}


//GLOBAL DISTRIBUTED AVG
public class GlobalDistributedAverage {

    public static void main(String[] args) {

        double[] values = { 10, 20, 30, 40 }; // each process value
        int n = values.length;

        // simulate multiple rounds
        for (int round = 1; round <= 5; round++) {

            double[] newValues = new double[n];

            for (int i = 0; i < n; i++) {

                double left = values[(i - 1 + n) % n];
                double right = values[(i + 1) % n];

                newValues[i] = (values[i] + left + right) / 3;
            }

            values = newValues;

            System.out.println("Round " + round + ":");
            for (int i = 0; i < n; i++) {
                System.out.print(values[i] + " ");
            }
            System.out.println();
        }
    }
}


//RICART
import java.util.*;

class Ricart {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of processes: ");
        int n = sc.nextInt();

        int[] ts = new int[n];

        System.out.println("Enter timestamps:");
        for (int i = 0; i < n; i++) {
            ts[i] = sc.nextInt();
        }

        System.out.print("Enter requesting process (0 to " + (n - 1) + "): ");
        int p = sc.nextInt();

        int replies = 0;

        System.out.println("\nP" + p + " requests CS");

        for (int i = 0; i < n; i++) {
            if (i == p)
                continue;

            System.out.println("P" + p + " -> REQUEST -> P" + i);

            if (ts[p] < ts[i] || (ts[p] == ts[i] && p < i)) {
                System.out.println("P" + i + " -> OK -> P" + p);
                replies++;
            } else {
                System.out.println("P" + i + " defers reply");
            }
        }

        if (replies == n - 1)
            System.out.println("\nP" + p + " enters CS");
        else
            System.out.println("\nP" + p + " waits for replies");

        sc.close();
    }
}

//LOAD BALANCING
import java.util.*;

public class LoadBalancing {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of nodes: ");
        int n = sc.nextInt();

        System.out.print("Enter number of processes: ");
        int p = sc.nextInt();

        int load[] = new int[n];

        for (int i = 0; i < p; i++) {

            int minNode = 0;

            for (int j = 1; j < n; j++) {
                if (load[j] < load[minNode]) {
                    minNode = j;
                }
            }

            load[minNode]++;

            System.out.println("Process " + (i + 1) + " assigned to Node " + minNode);
        }

        System.out.println("\nFinal Load Distribution:");

        for (int i = 0; i < n; i++) {
            System.out.println("Node " + i + " has " + load[i] + " processes");
        }
    }
}


//BULLY
import java.util.Scanner;

public class BullyAlgorithm {
    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);

        System.out.print("Enter number of processes: ");
        int n = sc.nextInt();

        int[] active = new int[n];

        // All processes active
        for (int i = 0; i < n; i++)
            active[i] = 1;

        System.out.print("Enter crashed process: ");
        int crash = sc.nextInt();
        active[crash] = 0;

        System.out.print("Enter initiator process: ");
        int init = sc.nextInt();

        int coordinator = init;

        // Election
        for (int i = init + 1; i < n; i++) {
            if (active[i] == 1) {
                System.out.println(init + " -> " + i);
                coordinator = i;
            }
        }

        System.out.println("Coordinator: " + coordinator);

        sc.close();
    }
}


//MULTITHREADING
class MyThread extends Thread {

    MyThread(String name) {
        super(name); // set thread name
    }

    public void run() {
        for (int i = 1; i <= 5; i++) {
            System.out.println(getName() + " running: " + i);
            try {
                Thread.sleep(500);
            } catch (Exception e) {
            }
        }
    }
}

public class MultithreadingDemo {
    public static void main(String[] args) {

        MyThread t1 = new MyThread("Thread-1");
        MyThread t2 = new MyThread("Thread-2");

        t1.start();
        t2.start();
    }
}


//IPC CLIENT
import java.net.*;
import java.io.*;
import java.util.Scanner;

public class Client {
    public static void main(String[] args) throws Exception {

        Socket socket = new Socket("localhost", 5000);

        Scanner in = new Scanner(socket.getInputStream());
        PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
        Scanner sc = new Scanner(System.in);

        while (true) {
            System.out.print("Client: ");
            String msg = sc.nextLine();
            out.println(msg);

            if (msg.isBlank())
                break;

            msg = in.nextLine();
            System.out.println("Server: " + msg);

            if (msg.isBlank())
                break;
        }

        socket.close();
    }
}

//IPC SERVER
import java.net.*;
import java.io.*;
import java.util.Scanner;

public class Server {
    public static void main(String[] args) throws Exception {

        ServerSocket ss = new ServerSocket(5000);
        System.out.println("Waiting...");

        Socket socket = ss.accept();
        System.out.println("Connected!");

        Scanner in = new Scanner(socket.getInputStream());
        PrintWriter out = new PrintWriter(socket.getOutputStream(), true);
        Scanner sc = new Scanner(System.in);

        while (true) {
            String msg = in.nextLine();
            System.out.println("Client: " + msg);

            if (msg.isBlank())
                break;

            System.out.print("Server: ");
            msg = sc.nextLine();
            out.println(msg);

            if (msg.isBlank())
                break;
        }

        socket.close();
        ss.close();
    }
}
