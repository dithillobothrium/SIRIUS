#include <sirius.h>

using namespace sirius;

void check_spline(Spline<double> const& s__, std::function<double(double)> f__, double x0, double x1)
{
    Radial_grid rgrid(linear_grid, 10000, x0, x1);
    for (int ir = 0; ir < 10000; ir++)
    {
        double x = rgrid[ir];
        if (std::abs(s__(x) - f__(x)) > 1e-10)
        {
            printf("wrong spline interpolation at x = %18.12f\n", x);
            printf("true value: %18.12f, spline value: %18.12f\n", f__(x), s__(x));
            exit(1);
        }
    }
}

void test_spline_1a()
{
    auto f = [](double x) { return std::sin(x) / x;}; 

    double x0 = 1e-7;
    double x1 = 2.0;
    
    Radial_grid rgrid(exponential_grid, 2000, 1e-7, 2);
    Spline<double> s(rgrid, f);

    check_spline(s, f, x0, x1);
    double v = s.integrate(0);

    if (std::abs(v - 1.605412876802697) > 1e-10)
    {
        printf("wrong integral\n");
        exit(1);
    }
}

void test_spline_1b()
{
    auto f = [](double x) { return std::exp(-2*x) * x;}; 

    double x0 = 1e-7;
    double x1 = 2.0;
    
    Radial_grid rgrid(exponential_grid, 2000, 1e-7, 2);
    Spline<double> s(rgrid, f);

    check_spline(s, f, x0, x1);
    double v = s.integrate(0);

    if (std::abs(v - 0.22710545138907753) > 1e-10)
    {
        printf("wrong integral\n");
        exit(1);
    }
}

void test_spline_3(std::vector< Spline<double> > const& s, std::function<double(double)> f__, double x0, double x1)
{
    for (int k = 0; k < 10; k++) check_spline(s[k], f__, x0, x1);
}

// Test vector of splines.
void test_spline_2()
{
    auto f = [](double x) { return std::sin(x) / x;}; 

    double x0 = 1e-7;
    double x1 = 2.0;

    Radial_grid rgrid(exponential_grid, 2000, 1e-7, 2);
    std::vector< Spline<double> > s(10);

    for (int k = 0; k < 10; k++) s[k] = Spline<double>(rgrid, f);
    test_spline_3(s, f, x0, x1);
}

// Test product of splines.
void test_spline_4()
{
    Radial_grid rgrid(exponential_grid, 2000, 1e-7, 4);
    Spline<double> s1(rgrid);
    Spline<double> s2(rgrid);
    Spline<double> s3(rgrid);

    for (int ir = 0; ir < 2000; ir++)
    {
        s1[ir] = std::sin(rgrid[ir] * 2) / rgrid[ir];
        s2[ir] = std::exp(rgrid[ir]);
        s3[ir] = s1[ir] * s2[ir];
    }
    s1.interpolate();
    s2.interpolate();
    s3.interpolate();

    Spline<double> s12 = s1 * s2;

    Radial_grid rlin(linear_grid, 20000, 1e-7, 4);
    double d = 0;
    for (int ir = 0; ir < rlin.num_points(); ir++)
    {
        double x = rlin[ir];
        d += std::pow(s3(x) - s12(x), 2);
    }
    d = std::sqrt(d / rlin.num_points());

    printf("RMS diff of spline product: %18.14f\n", d);

    if (d > 1e-6)
    {
        printf("wrong product of two splines\n");
        exit(1);
    }

    //FILE* fout = fopen("splne_prod.dat", "w");
    //Radial_grid rlin(linear_grid, 10000, 1e-7, 4);
    //for (int i = 0; i < rlin.num_points(); i++)
    //{
    //    double x = rlin[i];
    //    fprintf(fout, "%18.10f %18.10f %18.10f\n", x, s3(x), s12(x));
    //}
    //fclose(fout);
}

void test_spline_5()
{
    int N = 6000;
    int n = 256;

    Radial_grid rgrid(exponential_grid, N, 1e-7, 4);
    std::vector< Spline<double> > s1(n);
    std::vector< Spline<double> > s2(n);
    for (int i = 0; i < n; i++)
    {
        s1[i] = Spline<double>(rgrid);
        s2[i] = Spline<double>(rgrid);
        for (int ir = 0; ir < N; ir++)
        {
            s1[i][ir] = std::sin(rgrid[ir] * (1 + n * 0.01)) / rgrid[ir];
            s2[i][ir] = std::exp((1 + n * 0.01) * rgrid[ir]);
        }
        s1[i].interpolate();
        s2[i].interpolate();
    }
    mdarray<double, 2> prod(n, n);
    runtime::Timer t("spline|inner");
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            prod(i, j) = inner(s1[i], s2[j], 2);
        }
    }
    double tval = t.stop();
    DUMP("inner product time: %12.6f", tval);
    DUMP("performance: %12.6f GFlops", 1e-9 * n * n * N * 85 / tval);
}

void test_spline_6()
{
    mdarray<Spline<double>, 1> array(20);
    Radial_grid rgrid(exponential_grid, 300, 1e-7, 4);

    for (int i = 0; i < 20; i++)
    {
        array(i) = Spline<double>(rgrid);
        for (int ir = 0; ir < rgrid.num_points(); ir++) array(i)[ir] = std::exp(-rgrid[ir]);
        array(i).interpolate();
    }
}

void test1(double x0, double x1, int m, double exact_result)
{
    printf("\n");
    printf("test1: integrate sin(x) * x^{%i} and compare with exact result\n", m);
    printf("       lower and upper boundaries: %f %f\n", x0, x1);
    Radial_grid r(exponential_grid, 5000, x0, x1);
    Spline<double> s(r);
    
    for (int i = 0; i < 5000; i++) s[i] = std::sin(r[i]);
    
    double d = s.interpolate().integrate(m);
    double err = std::abs(1 - d / exact_result);
    
    printf("       relative error: %18.12f", err);
    if (err < 1e-10) 
    {
        printf("  OK\n");
    }
    else
    {
        printf("  Fail\n");
        exit(1);
    }
}

void test2(radial_grid_t grid_type, double x0, double x1)
{
    printf("\n");
    printf("test2: value and derivatives of exp(x)\n");

    int N = 5000;
    Radial_grid r(grid_type, N, x0, x1);
    Spline<double> s(r, [](double x){return std::exp(x);});
    
    printf("grid type : %s\n", r.grid_type_name().c_str());

    //== std::string fname = "grid_" + r.grid_type_name() + ".txt";
    //== FILE* fout = fopen(fname.c_str(), "w");
    //== for (int i = 0; i < r.num_points(); i++) fprintf(fout,"%i %16.12e\n", i, r[i]);
    //== fclose(fout);
    
    printf("x = %f, true exp(x) = %f, exp(x) = %f, exp'(x)= %f, exp''(x) = %f\n", x0, s[0], s.deriv(0, 0), s.deriv(1, 0), s.deriv(2, 0));
    printf("x = %f, true exp(x) = %f, exp(x) = %f, exp'(x)= %f, exp''(x) = %f\n", x1, s[N - 1], s.deriv(0, N - 1), s.deriv(1, N - 1), s.deriv(2, N - 1));
}

void test3(int m, double x0, double x1, double exact_val)
{
    printf("\n");
    printf("test3\n");
    
    Radial_grid r(exponential_grid, 2000, x0, x1);
    Spline<double> s1(r);
    Spline<double> s2(r);
    Spline<double> s3(r);

    for (int i = 0; i < 2000; i++)
    {
        s1[i] = std::sin(r[i]) / r[i];
        s2[i] = std::exp(-r[i]) * std::pow(r[i], 8.0 / 3.0);
        s3[i] = s1[i] * s2[i];
    }
    s1.interpolate();
    s2.interpolate();
    s3.interpolate();

    double v1 = s3.integrate(m);
    double v2 = inner(s1, s2, m);

    printf("interpolate product of two functions and then integrate with spline   : %16.12f\n", v1);
    printf("interpolate two functions and then integrate the product analytically : %16.12f\n", v2);
    printf("                                                           difference : %16.12f\n", std::abs(v1 - v2));
    printf("                                                         exact result : %16.12f\n", exact_val);

    if (std::abs(v1 - v2) > 1e-10)
    {
        printf("wrong inner product of splines\n");
        exit(1);
    }
}

void test5()
{
    printf("\n");
    printf("test5: high-order integration\n");

    int N = 2000;
    Radial_grid r(exponential_grid, N, 1e-8, 0.9);
    Spline<double> s(r, [](double x){return std::log(x);});
    double true_value = -0.012395331058672921;
    if (std::abs(s.integrate(7) - true_value) > 1e-10)
    {
        printf("wrong high-order integration\n");
        exit(1);
    }
    else
    {
        printf("OK\n");
    }
}

void test6()
{
    printf("\n");
    printf("test6: 4 points interpolation\n");
    std::vector<double> x = {0, 1, 2, 3};
    Radial_grid r(x);
    Spline<double> s(r);
    s[0] = 0;
    s[1] = 1;
    s[2] = 0;
    s[3] = 0;
    double val = s.interpolate().integrate(0);
    if (std::abs(val - 1.125) > 1e-13)
    {
        printf("wrong result: %18.12f\n", val);
        exit(1);
    }
    else
    {
        printf("OK\n");
    }
    
    //== int N = 4000;
    //== FILE* fout = fopen("spline_test.dat", "w");
    //== for (int i = 0; i < N; i++)
    //== {
    //==     double t = 3.0 * i / (N - 1);
    //==     fprintf(fout, "%18.12f %18.12f\n", t, s(t));
    //== }
    //== fclose(fout);
}

int main(int argn, char** argv)
{
    sirius::initialize(1);

    test_spline_1a();
    test_spline_1b();
    test_spline_2();
    test_spline_4();
    test_spline_5();
    test_spline_6();

    test1(0.1, 7.13, 0, 0.3326313127230704);
    test1(0.1, 7.13, 1, -3.973877090504168);
    test1(0.1, 7.13, 2, -23.66503552796384);
    test1(0.1, 7.13, 3, -101.989998166403);
    test1(0.1, 7.13, 4, -341.6457111811293);
    test1(0.1, 7.13, -1, 1.367605245879218);
    test1(0.1, 7.13, -2, 2.710875755556171);
    test1(0.1, 7.13, -3, 9.22907091561693);
    test1(0.1, 7.13, -4, 49.40653515725798);
    test1(0.1, 7.13, -5, 331.7312413927384);

    double x0 = 0.00001;
    test2(linear_grid, x0, 2.0);
    test2(exponential_grid, x0, 2.0);
    test2(scaled_pow_grid, x0, 2.0);
    test2(pow2_grid, x0, 2.0);
    test2(pow3_grid, x0, 2.0);

    test3(1, 0.0001, 2.0, 0.7029943796175838);
    test3(2, 0.0001, 2.0, 1.0365460153117974);

    test5();

    test6();

    sirius::finalize();
    
    return 0;
}