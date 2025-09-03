//
// A8_Reduce en OpenMP.cpp
/*
Integrantes:
- Navarro González Samantha
- Parra Lopez Maria Fernanda
- Ruiz Mejorada Diego
Actividad 7: Modelo de Memoria en OpenMP
*/

#include <omp.h>
#include <iostream>
#include <iomanip>
#include <random>
#include <climits>
#include <ctime>
#include <limits>
using namespace std;

class OperacionesVectoriales {
public:
    int* crearVector(int n) { return new int[n]; }
    void liberarVector(int* v) { delete[] v; }

    void llenarAscendente(int* v, int n) {
#pragma omp parallel for
        for (int i = 0; i < n; ++i) v[i] = i + 1;
    }

    // Llenado aleatorio thread-safe (motor por hilo)
    void llenarAleatorioOpenMP(int* v, int n, int minVal = 1, int maxVal = 2'000'000) {
        std::uniform_int_distribution<int> dist(minVal, maxVal);
#pragma omp parallel
        {
            std::seed_seq seed{
                (unsigned)time(nullptr),
                (unsigned)omp_get_thread_num(),
                0x9e3779b9u
            };
            std::mt19937 rng(seed);

#pragma omp for
            for (int i = 0; i < n; ++i) v[i] = dist(rng);
        }
    }

    // ===== Operaciones con reduce =====
    long long sumatoriaOpenMP(const int* v, int n) {
        long long s = 0;
#pragma omp parallel for reduction(+:s)
        for (int i = 0; i < n; ++i) s += v[i];
        return s;
    }

    double promedioOpenMP(const int* v, int n) {
        long long s = sumatoriaOpenMP(v, n);
        return (double)s / (double)n;
    }

    int maximoOpenMP(const int* v, int n) {
        int mx = INT_MIN;
#pragma omp parallel for reduction(max:mx)
        for (int i = 0; i < n; ++i) mx = v[i] > mx ? v[i] : mx;
        return mx;
    }

    int minimoOpenMP(const int* v, int n) {
        int mn = INT_MAX;
#pragma omp parallel for reduction(min:mn)
        for (int i = 0; i < n; ++i) mn = v[i] < mn ? v[i] : mn;
        return mn;
    }

    // ===== Helpers de impresión (formato 10x10) =====
    void imprimirComoMatriz10x10(const int* v) {
        for (int r = 0; r < 10; ++r) {
            for (int c = 0; c < 10; ++c) {
                cout << setw(7) << v[r * 10 + c] << ' ';
            }
            cout << '\n';
        }
    }
};

// ---- Demos pedidas ----
void demo10x10(OperacionesVectoriales& op) {
    cout << "\n=== DEMO 1: 10x10 (vector de 100 ascendente) ===\n";
    const int n = 100;
    int* A = op.crearVector(n);
    op.llenarAscendente(A, n);

    cout << "Vector A mostrado como 10x10 (antes de operar):\n";
    op.imprimirComoMatriz10x10(A);

    // Sumatoria
    double t0 = omp_get_wtime();
    long long S = op.sumatoriaOpenMP(A, n);
    double t1 = omp_get_wtime();
    cout << "\nSumatoria: " << S << "  | Tiempo: " << (t1 - t0) << " s\n";

    // Promedio
    t0 = omp_get_wtime();
    double P = op.promedioOpenMP(A, n);
    t1 = omp_get_wtime();
    cout << "Promedio: " << P << "  | Tiempo: " << (t1 - t0) << " s\n";

    // Máximo
    t0 = omp_get_wtime();
    int MX = op.maximoOpenMP(A, n);
    t1 = omp_get_wtime();
    cout << "Maximo: " << MX << "  | Tiempo: " << (t1 - t0) << " s\n";

    // Mínimo
    t0 = omp_get_wtime();
    int MN = op.minimoOpenMP(A, n);
    t1 = omp_get_wtime();
    cout << "Minimo: " << MN << "  | Tiempo: " << (t1 - t0) << " s\n";

    op.liberarVector(A);
}

void demo1000x1000(OperacionesVectoriales& op) {
    cout << "\n=== DEMO 2: 1000x1000 (vector de 1,000,000 aleatorio) ===\n";
    const int filas = 1000, cols = 1000, n = filas * cols;

    int* A = op.crearVector(n);

    double t0 = omp_get_wtime();
    op.llenarAleatorioOpenMP(A, n);
    double t1 = omp_get_wtime();
    cout << "Llenado aleatorio: " << (t1 - t0) << " s\n";

    t0 = omp_get_wtime();
    long long S = op.sumatoriaOpenMP(A, n);
    t1 = omp_get_wtime();
    cout << "Sumatoria: " << (t1 - t0) << " s\n";

    t0 = omp_get_wtime();
    double P = (double)S / (double)n; // reusa S para ahorrar una pasada
    t1 = omp_get_wtime();
    cout << "Promedio (post-proc): " << (t1 - t0) << " s  | valor=" << P << "\n";

    t0 = omp_get_wtime();
    int MX = op.maximoOpenMP(A, n);
    t1 = omp_get_wtime();
    cout << "Maximo: " << (t1 - t0) << " s  | valor=" << MX << "\n";

    t0 = omp_get_wtime();
    int MN = op.minimoOpenMP(A, n);
    t1 = omp_get_wtime();
    cout << "Minimo: " << (t1 - t0) << " s  | valor=" << MN << "\n";

    op.liberarVector(A);
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    OperacionesVectoriales op;

    // Estado de trabajo por defecto: 10x10 (vector de 100)
    int filas = 10, cols = 10;
    int n = filas * cols;
    int* V = op.crearVector(n);
    op.llenarAscendente(V, n); // arranque con ascendente

    int opcion;
    do {
        cout << "\n--- MENU (OpenMP reduce) ---\n";
        cout << "1) Sumatoria del vector\n";
        cout << "2) Promedio del vector\n";
        cout << "3) Maximo del vector\n";
        cout << "4) Minimo del vector\n";
        cout << "5) Llenar vector ascendente (tam actual)\n";
        cout << "6) Llenar vector aleatorio (tam actual)\n";
        cout << "7) DEMO 1 (10x10 ascendente: imprime y tiempos)\n";
        cout << "8) DEMO 2 (1000x1000 aleatorio: solo tiempos)\n";
        cout << "9) Cambiar tamano (ej. 100 -> 1000000) \n";
        cout << "0) Salir\n";
        cout << "Opcion: ";
        if (!(cin >> opcion)) break;

        double t0, t1;
        switch (opcion) {
        case 1: {
            if (n == 100) { cout << "\nVector (10x10) antes:\n"; op.imprimirComoMatriz10x10(V); }
            t0 = omp_get_wtime();
            long long s = op.sumatoriaOpenMP(V, n);
            t1 = omp_get_wtime();
            cout << "Sumatoria = " << s << " | Tiempo: " << (t1 - t0) << " s\n";
        } break;

        case 2: {
            if (n == 100) { cout << "\nVector (10x10) antes:\n"; op.imprimirComoMatriz10x10(V); }
            t0 = omp_get_wtime();
            double p = op.promedioOpenMP(V, n);
            t1 = omp_get_wtime();
            cout << "Promedio = " << p << " | Tiempo: " << (t1 - t0) << " s\n";
        } break;

        case 3: {
            if (n == 100) { cout << "\nVector (10x10) antes:\n"; op.imprimirComoMatriz10x10(V); }
            t0 = omp_get_wtime();
            int mx = op.maximoOpenMP(V, n);
            t1 = omp_get_wtime();
            cout << "Maximo = " << mx << " | Tiempo: " << (t1 - t0) << " s\n";
        } break;

        case 4: {
            if (n == 100) { cout << "\nVector (10x10) antes:\n"; op.imprimirComoMatriz10x10(V); }
            t0 = omp_get_wtime();
            int mn = op.minimoOpenMP(V, n);
            t1 = omp_get_wtime();
            cout << "Minimo = " << mn << " | Tiempo: " << (t1 - t0) << " s\n";
        } break;

        case 5: {
            op.llenarAscendente(V, n);
            if (n == 100) { cout << "Vector ascendente (10x10):\n"; op.imprimirComoMatriz10x10(V); }
        } break;

        case 6: {
            int a = 1, b = 2'000'000;
            cout << "Rango aleatorio [min max] (default 1 2000000): ";
            if (!(cin >> a >> b)) { cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n'); a = 1; b = 2'000'000; }
            t0 = omp_get_wtime();
            op.llenarAleatorioOpenMP(V, n, a, b);
            t1 = omp_get_wtime();
            cout << "Vector aleatorio listo. Tiempo: " << (t1 - t0) << " s\n";
            if (n == 100) { op.imprimirComoMatriz10x10(V); }
        } break;

        case 7: demo10x10(op); break;
        case 8: demo1000x1000(op); break;

        case 9: {
            cout << "Nuevo tamano (ej. 100 -> 1,000,000): ";
            int nuevoN;
            if (cin >> nuevoN && nuevoN > 0) {
                op.liberarVector(V);
                n = nuevoN;
                V = op.crearVector(n);
                // por default, lo dejamos ascendente para que las ops tengan datos:
                op.llenarAscendente(V, n);
                cout << "Tamano actualizado a " << n << ".\n";
            }
            else {
                cout << "Tamano invalido.\n";
                cin.clear(); cin.ignore(numeric_limits<streamsize>::max(), '\n');
            }
        } break;

        case 0: cout << "Adios.\n"; break;
        default: cout << "Opcion invalida.\n";
        }
    } while (opcion != 0);

    op.liberarVector(V);
    return 0;
}

