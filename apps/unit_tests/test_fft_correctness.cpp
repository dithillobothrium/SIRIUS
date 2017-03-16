#include <sirius.h>

using namespace sirius;

void test_fft(double cutoff__)
{
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    FFT3D fft(find_translations(cutoff__, M), mpi_comm_world(), CPU);

    Gvec gvec(M, cutoff__, mpi_comm_world(), mpi_comm_world(), false);

    if (mpi_comm_world().rank() == 0) {
        printf("num_gvec: %i\n", gvec.num_gvec());
    }
    MPI_grid mpi_grid(mpi_comm_world());

    printf("num_gvec_fft: %i\n", gvec.partition().gvec_count_fft());
    printf("offset_gvec_fft: %i\n", gvec.partition().gvec_offset_fft());

    fft.prepare(gvec.partition());

    mdarray<double_complex, 1> f(gvec.num_gvec());
    for (int ig = 0; ig < gvec.num_gvec(); ig++) {
        auto v = gvec.gvec(ig);
        if (mpi_comm_world().rank() == 0) {
            printf("ig: %6i, gvec: %4i %4i %4i   ", ig, v[0], v[1], v[2]);
        }
        f.zero();
        f[ig] = 1.0;
        fft.transform<1>(gvec.partition(), &f[gvec.partition().gvec_offset_fft()]);

        double diff = 0;
        /* loop over 3D array (real space) */
        for (int j0 = 0; j0 < fft.grid().size(0); j0++) {
            for (int j1 = 0; j1 < fft.grid().size(1); j1++) {
                for (int j2 = 0; j2 < fft.local_size_z(); j2++) {
                    /* get real space fractional coordinate */
                    auto rl = vector3d<double>(double(j0) / fft.grid().size(0), 
                                               double(j1) / fft.grid().size(1), 
                                               double(fft.offset_z() + j2) / fft.grid().size(2));
                    int idx = fft.grid().index_by_coord(j0, j1, j2);

                    diff += std::pow(std::abs(fft.buffer(idx) - std::exp(double_complex(0.0, twopi * (rl * v)))), 2);
                }
            }
        }
        mpi_comm_world().allreduce(&diff, 1);
        diff = std::sqrt(diff / fft.size());
        if (mpi_comm_world().rank() == 0) {
            printf("error : %18.10e", diff);
            if (diff < 1e-10) {
                printf("  OK\n");
            } else {
                printf("  Fail\n");
                exit(1);
            }
        }
    }

    fft.dismiss();
}

int main(int argn, char **argv)
{
    cmd_args args;
    args.register_key("--cutoff=", "{double} cutoff radius in G-space");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }

    double cutoff = args.value<double>("cutoff", 5);

    sirius::initialize(1);

    test_fft(cutoff);

    sddk::timer::print();
    
    sirius::finalize();
    return 0;
}
