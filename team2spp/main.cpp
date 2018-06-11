#define __CL_ENABLE_EXCEPTIONS
#include<iostream>
#include<unordered_map>
#include<cstdio>
#include<cstdlib>
#include<random>
#include<CL/cl.hpp>
#include<Windows.h>

#define MSIZE 2000
#define MAX_EDGES 10
#define D_INF 0x0FFFFFFF
#define USE_GPU

class Node {
private:
	int nodenum = 0;
	std::unordered_map<const Node*, int> edges;
public:
	int getId() const {
		return nodenum;
	}
	void setNodenumber(int idx) {
		nodenum = idx;
	}
	void addEdges(const Node *n, int weight) {
		if (n == this) return;
		edges.insert(std::pair<const Node*, int>(n, weight));
	}
	int to(Node *n) {
		try {
			return edges.at(n);
		}
		catch (std::out_of_range) {
			return D_INF;
		}
	}

	size_t totalEdges() {
		return edges.size();
	}

	bool operator == (const Node &n) const {
		return nodenum == n.nodenum;
	}

	friend std::ostream& operator<<(std::ostream &os, const Node &n) {
		os << "node: " << n.nodenum << "\n";
		for (auto it = n.edges.begin(); it != n.edges.end(); it++) {
			std::cout << " [" << it->first->nodenum << ", " << it->second << "]";
		}
		return os;
	}
};


void floyd(int d[], int p[]) {
	for (int k = 0; k < MSIZE; k++) {
		for (int i = 0; i < MSIZE; i++) {
			for (int j = 0; j < MSIZE; j++) {
				int dk = d[i*MSIZE + k] + d[k*MSIZE + j];
				if (dk < d[i*MSIZE + j]) {
					p[i*MSIZE + j] = k;
					d[i*MSIZE + j] = dk;
				}
			}
		}
	}
}

void node_djikstra(Node n[], int touch[]) {
	int i, vnear;
	int min;
	int length[MSIZE];
	touch[0] = -1;
	for (i = 1; i < MSIZE; i++) {
		touch[i] = 0;
		length[i] = n[0].to(&n[i]);
	}
	for (int x = 1; x < MSIZE; x++) {
		min = INFINITE;
		for (i = 1; i < MSIZE; i++) {
			if (length[i] >= 0 && length[i] < min) {
				min = length[i];
				vnear = i;
			}
		}
		for (i = 1; i < MSIZE; i++) {
			int calclen = length[vnear] + n[vnear].to(&n[i]);
			if (calclen < length[i]) {
				length[i] = calclen;
				touch[i] = vnear;
			}
		}
		length[vnear] = -1;
	}
}

void djikstra(int *d, int touch[]) {
	int i, vnear;
	int min;
	int length[MSIZE];
	touch[0] = -1;
	for (i = 1; i < MSIZE; i++) {
		touch[i] = 0;
		length[i] = d[i];
	}
	for (int x = 1; x < MSIZE; x++) {
		min = D_INF;
		for (i = 1; i < MSIZE; i++) {
			if (length[i] >= 0 && length[i] < min) {
				min = length[i];
				vnear = i;
			}
		}
		for (i = 1; i < MSIZE; i++) {
			int calclen = length[vnear] + d[MSIZE*vnear + i];
			if (calclen < length[i]) {
				length[i] = calclen;
				touch[i] = vnear;
			}
		}
		length[vnear] = -1;
	}
}

void randomnode(Node n[]) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> udist(1, MSIZE);
	for (int x = 0; x < MSIZE; x++) {
		for (int y = 0; y<MAX_EDGES;) {
			int i = udist(gen) - 1;
			if (x == i) continue;
			n[x].addEdges(&n[i], udist(gen));
			y++;
		}
	}
}

void randompath(int *w) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> udist(1, 999);
	for (int x = 0; x < MSIZE; x++) {
		for(int y=0;y<MSIZE;y++){
			if (x == y)w[x*MSIZE + y] = 0;
			else w[x*MSIZE+y] = udist(gen);
		}
	}
}
std::vector<int> pathVerifier;
void path(const int touch[], int dst) {
	int prev = touch[dst];
	if (prev < 0) {
		std::cout << "node0 ";
		pathVerifier.push_back(0);
		return;
	}
	path(touch, prev);
	std::cout << "node" << dst << " ";
	pathVerifier.push_back(dst);
}

void floyd_path(const int mat[], int x, int y) {
	if (mat[x*MSIZE + y] == -1) { return; }
	floyd_path(mat, x, mat[x*MSIZE + y]);
	std::cout << "node" << mat[x*MSIZE + y] << " ";
	pathVerifier.push_back(mat[x*MSIZE + y]);
	floyd_path(mat, mat[x*MSIZE + y], y);
}
void path(const int mat[], int x, int y) {
	std::cout << "node" << x << " ";
	pathVerifier.push_back(x);
	floyd_path(mat, x, y);
	std::cout << "node" << y;
	pathVerifier.push_back(y);
}

int main(int argc, char* argv[]) {
	cl_int err = CL_SUCCESS;
	std::cout << "testing on node count: " << MSIZE << std::endl;
	std::cout.fill('0');
	try {
#ifdef USE_GPU
		std::vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		if (platforms.size() == 0) {
			std::cout << "Platform size 0\n";
			return -1;

		}
		cl::Platform default_platform = platforms[0];
		std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";
		cl_context_properties properties[] =
		{ CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
		cl::Context context(CL_DEVICE_TYPE_GPU, properties);

		std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
		cl::Device default_device = devices[0];
		std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";
		

		std::string kernel_code =
			"   __kernel void floyd(global int* d, global int* p, const int n, const int k) {"
			"       int i, j;"
			""
			"       i = get_global_id(1);"
			"       j = get_global_id(0);"
			"		int dk = d[i*n+k]+d[k*n+j];"
			"       if( dk < d[i*n+j] ) {"
			"			p[i*n+j] = k;"
			"			d[i*n+j] = dk;"
			"		}"
			"   }";
			

		cl::Program::Sources source(1,
			std::make_pair(kernel_code.c_str(), kernel_code.length()));

		cl::Program program_ = cl::Program(context, source);

		program_.build(devices);
		int n = MSIZE;

		// create buffers on device (allocate space on GPU)
		cl::Buffer buffer_d(context, CL_MEM_READ_WRITE, sizeof(int) * n* n);
		cl::Buffer buffer_p(context, CL_MEM_READ_WRITE, sizeof(int) * n* n);

			
		int *w = new int[MSIZE*MSIZE];
		int *touch = new int[MSIZE];
		int *p = new int[MSIZE*MSIZE];
		randompath(w);

		memset(p, -1, sizeof(int)*n*n);
		
		//run djikstra cpu first
		ULONGLONG delta = GetTickCount64();
		djikstra(w, touch);
		delta = GetTickCount64() - delta;
		printf("ended in %d.%03d\n", delta / 1000, delta % 1000);
		path(touch, MSIZE - 1);
		std::cout << std::endl;
		int totalweight = 0;
		for (int i = 0; i < pathVerifier.size()-1; i++) {
			int src = pathVerifier.at(i);
			int dst = pathVerifier.at(i + 1);
			int weight = w[src*MSIZE + dst];
			std::cout << "(" << src << "," << dst << ")" << "=" << weight<<std::endl;
			totalweight += weight;
		}
		std::cout << totalweight << std::endl;

		cl::CommandQueue queue(context, default_device);

		// push write commands to queue
		queue.enqueueWriteBuffer(buffer_d, CL_TRUE, 0, sizeof(int)*n*n, w);
		queue.enqueueWriteBuffer(buffer_p, CL_TRUE, 0, sizeof(int)*n*n, p);
		cl::Kernel kernel(program_, "floyd");
		kernel.setArg(0, buffer_d);
		kernel.setArg(1, buffer_p);
		kernel.setArg(2, n);

		delta = GetTickCount64();
		//cl::Event qevent;
		for(int k=0;k<n;k++){
			kernel.setArg(3, k);
			queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(MSIZE, MSIZE), cl::NDRange(10, 10), NULL);
		}
		//qevent.wait();
		delta = GetTickCount64() - delta;
		printf("ended in %d.%03d\n", delta / 1000, delta % 1000);
		queue.enqueueReadBuffer(buffer_p, CL_TRUE, 0, sizeof(int)*n*n, p);
		pathVerifier.clear();
		path(p, 0, MSIZE - 1);
		std::cout << std::endl;
		totalweight = 0;
		for (int i = 0; i < pathVerifier.size() - 1; i++) {
			int src = pathVerifier.at(i);
			int dst = pathVerifier.at(i + 1);
			int weight = w[src*MSIZE + dst];
			std::cout << "(" << src << "," << dst << ")" << "=" << weight << std::endl;
			totalweight += weight;
		}
		std::cout << totalweight << std::endl;
		queue.flush();
#endif

		/*Node *nodes = new Node[MSIZE];
		for (int i = 0; i < MSIZE; i++) {
			nodes[i].setNodenumber(i);
		}
		randomnode(nodes);
		node_djikstra(nodes, touch);

		for (int i = 0; i<MSIZE; i++) {
			for(int x=0; x<MSIZE; x++){
				d[i*MSIZE + x] = nodes[i].to(nodes+x);
				p[i*MSIZE + x] = -1;
			}
		}
		*/
#ifndef USE_GPU
		int *w = new int[MSIZE*MSIZE];
		int *touch = new int[MSIZE];

		int *d = new int[MSIZE*MSIZE];
		int *p = new int[MSIZE*MSIZE];
		randompath(w);
		for (int i = 0; i<MSIZE*MSIZE; i++) {
				d[i] = w[i];
				p[i] = -1;
		}

		ULONGLONG delta = GetTickCount64();
		djikstra(w, touch);
		delta = GetTickCount64() - delta;
		
		printf("ended in %d.%03d\n", delta / 1000 ,delta % 1000);
		path(touch, MSIZE - 1);
		std::cout << std::endl;
		delta = GetTickCount64();
		floyd(d, p);
		delta = GetTickCount64() - delta;
		printf("ended in %d.%03d\n", delta / 1000, delta % 1000);
		path(p, 0, MSIZE-1);
		std::cout << std::endl;
		/*
		for (int i = 0; i < MSIZE; i++) {
			std::cout << nodes[i] << std::endl;
		}*/
#endif
	}
	catch (cl::Error err) {
		std::cerr
			<< "ERROR: "
			<< err.what()
			<< "("
			<< err.err()
			<< ")"
			<< std::endl;

	}
	std::cout << "press return to exit...";
	getchar();
	return EXIT_SUCCESS;
}