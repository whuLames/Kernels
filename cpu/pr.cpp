/*
实现Pagerank的并行和非并行版本
*/
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
using namespace std;
typedef pair<int, int> PII;

struct graph {
    int m, n;
    void init();
};

bool isConvergence(vector<float>& last, vector<float>& now, float eps)
{
    int n = last.size();
    // for(int i = 0; i < n; i ++) {
    //     float a = abs(last[i] - now[i]);
    //     cout << "The abs of node :" << i << " is " << a << " ";
    // }
    // cout << endl;
    for(int i = 0; i < n; i ++) {
        if(abs(last[i] - now[i]) >= eps) return false;
    }
    return true;
}

void prCompute(vector<int>& head, vector<int>& next, vector<int>&degrees, vector<int>& e, vector<float>& last, vector<float>& now, float dampling_factor)
{
    int n = last.size();
    // for(int i = 0; i < n; i ++) last[i] = now[i];
    last.assign(now.begin(), now.end());
    now.assign(n, 0.0);
    
    float value_zero_deg_node = 0;
    for(int i = 0; i < n; i ++) {
        // float val = last[i] / degrees[i];
        // cout << "node " << i << "have degrees: " << degrees[i] << " pr value: " << last[i] << " send pr value: " << val << endl;
        if(degrees[i] == 0) value_zero_deg_node += last[i];
        else {
            float val = last[i] / degrees[i];
            for(int j = head[i]; ~j; j = next[j]) {
                int node = e[j];
                now[node] += val;
            }
        }
    }

    for(int i = 0; i < n; i ++) {
        // now[i] = now[i] * dampling_factor + (1 - dampling_factor + value_zero_deg_node) / n;
        now[i] += value_zero_deg_node / n;
        now[i] = now[i] * dampling_factor + (1 - dampling_factor) / n;
    }

}

void pagerank(int n, vector<PII>& edges, int rounds, float eps, float dampling_factor)
{
    int m = edges.size();
    vector<int> head(n, -1); //
    vector<int> next(m, -1); 
    vector<int> e(m, -1);
    vector<int> degrees(n, 0);
    int idx = 0;

    // build the graph
    for(int i = 0; i < m; i ++) {
        int a = edges[i].first, b = edges[i].second;
        e[idx] = b;
        next[idx] = head[a];
        head[a] = idx ++;
        degrees[a] ++;
    }
    vector<float> pr_last(n, float(1.0)/n); 
    vector<float> pr_now;
    pr_now.assign(pr_last.begin(), pr_last.end());
    int r = 0;

    do {
        float sum = 0;
        cout << "Single pr do " << r << "round " << endl;
        for(int i = 0; i < n; i ++) {
            cout << "The pr value of vertex: " << i << " is :" << pr_now[i] << "\n";
            sum += pr_now[i];
        }
        // cout << "The sum in round: " << r << " is: " << sum << endl;
        prCompute(head, next, degrees, e, pr_last, pr_now, dampling_factor);
        r ++;
    } while(!isConvergence(pr_last, pr_now, eps) && r < rounds);

    for(int i = 0; i < n; i ++) {
        cout << "The pr value of vertex: " << i << " is :" << pr_now[i] << endl;
    }

}


void vertexCompute(int vertex_id, vector<float>& last, vector<float>& now, vector<int>& head, vector<int>& next, vector<int>& e, vector<int>& degrees)
{
    // a thread for a vertex
    static mutex mtx;
    // unique_lock<mutex> lock(mtx); // 构造函数会自动调用lock
    // lock.unlock(); 
    
    int n = now.size();
    
    // float val = now[vertex_id];
    if(degrees[vertex_id] == 0) {
        float val = last[vertex_id] / n;
        for(int i = 0; i < n; i ++) {
            // 上锁
            unique_lock<mutex> lock(mtx);
            now[i] += val;
            lock.unlock();
        }
    } else {
        float val = last[vertex_id] / degrees[vertex_id];
        for(int i = head[vertex_id]; ~i; i = next[i]) {
            int node = e[i];
            unique_lock<mutex> lock(mtx);
            now[node] += val;
            lock.unlock();
        }
    }
}
void pagerank_multithreads(int n, vector<PII>& edges, int rounds, float eps, float dampling_factor)
{
    cout << "rounds: " << rounds << endl;
    // return;
    int m = edges.size();
    vector<int> head(n, -1); 
    vector<int> next(m, -1); 
    vector<int> e(m, -1);
    vector<int> degrees(n, 0);
    int idx = 0;

    // build the graph
    for(int i = 0; i < m; i ++) {
        int a = edges[i].first, b = edges[i].second;
        e[idx] = b;
        next[idx] = head[a];
        head[a] = idx ++;
        degrees[a] ++;
    }

    vector<float> pr_last(n, float(1.0)/n); 
    vector<float> pr_now;
    pr_now.assign(pr_last.begin(), pr_last.end());
    int r = 0;
    /*
    A thread for a vertex in push model
    */
    
    // vertexCompute(int vertex_id, vector<int>& last, vector<float>& now, vector<int>& head, vector<int>& next, vector<int>& e, vector<int>& degrees)

    do {
        cout << "Round: " << r << " do computation" << endl; 
        pr_last.assign(pr_now.begin(), pr_now.end());
        pr_now.assign(n, 0.0f);
        thread workThreads[n];
        for(int i = 0; i < n; i ++) {
            workThreads[i] = thread(vertexCompute, i, ref(pr_last), ref(pr_now), ref(head), ref(next), ref(e), ref(degrees));
        }
        for(int i = 0; i < n; i ++) workThreads[i].join();

        for(int i = 0; i < n; i ++) {
            pr_now[i] = pr_now[i] * 0.85 + (1-dampling_factor) / n;
        }

        float sum = 0.0;
        for(int i = 0; i < n; i ++) {
            sum += pr_now[i];
            printf("The value of vertex %d is %f \n", i, pr_now[i]);
        }
        printf("sum is %f \n", sum);
        r ++;
    } while(r < rounds && !isConvergence(pr_last, pr_now, eps));

    
}   

int main(int argc, char** argv)
{
    int fuc_id = stoi(argv[1]);
    int n; // the num of nodes;
    int m; // the num of edges
    cin >> n >> m;
    vector<PII> edges;
    cout << "The number of nodes: " << n << " The number of edges: " << m << endl;
    int a, b;
    while (m--)
    {
        cin >> a >> b;
        cout << "an edge from: " << a << " to " << b << endl;
        edges.push_back({a, b});
    }
    
    int rounds;
    float eps, damping_factor;
    cin >> rounds >> eps >> damping_factor;
    cout << "rounds: " << rounds << " eps: " << eps << " factor: " << damping_factor << endl;
    
    if(fuc_id == 0) pagerank(n, edges, rounds, eps, damping_factor);
    else if(fuc_id == 1) pagerank_multithreads(n, edges, rounds, eps, damping_factor);
    else if(fuc_id == 2) {
        pagerank(n, edges, rounds, eps, damping_factor);
        pagerank_multithreads(n, edges, rounds, eps, damping_factor);
    }
}