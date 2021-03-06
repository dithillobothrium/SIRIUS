#include <sirius.h>

using namespace sirius;

void test_hloc(std::vector<int> mpi_grid_dims__, double cutoff__, int num_bands__, int reduce_gvec__,
               int use_gpu__, int gpu_ptr__)
{
    device_t pu = static_cast<device_t>(use_gpu__);

    MPI_grid mpi_grid(mpi_grid_dims__, mpi_comm_world()); 
    
    matrix3d<double> M = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

    FFT3D_grid fft_box(find_translations(2.01 * cutoff__, M));

    FFT3D fft(fft_box, mpi_grid.communicator(1 << 0), pu);

    Gvec gvec(M, cutoff__, mpi_comm_world(), mpi_grid.communicator(1 << 0), reduce_gvec__);

    if (mpi_comm_world().rank() == 0) {
        printf("total number of G-vectors: %i\n", gvec.num_gvec());
        printf("local number of G-vectors: %i\n", gvec.gvec_count(0));
        printf("FFT grid size: %i %i %i\n", fft_box.size(0), fft_box.size(1), fft_box.size(2));
        printf("number of FFT threads: %i\n", omp_get_max_threads());
        printf("number of FFT groups: %i\n", mpi_grid.dimension_size(1));
        printf("MPI grid: %i %i\n", mpi_grid.dimension_size(0), mpi_grid.dimension_size(1));
        printf("number of z-columns: %i\n", gvec.num_zcol());
    }

    fft.prepare(gvec.partition());

    Simulation_parameters params;
    params.set_processing_unit(pu);
    
    Local_operator hloc(params, fft, gvec);

    Wave_functions phi(pu, gvec, 4 * num_bands__, 1);
    for (int i = 0; i < 4 * num_bands__; i++) {
        for (int j = 0; j < phi.component(0).pw_coeffs().num_rows_loc(); j++) {
            phi.component(0).pw_coeffs().prime(j, i) = type_wrapper<double_complex>::random();
        }
        phi.component(0).pw_coeffs().prime(0, i) = 1.0;
    }
    Wave_functions hphi(pu, gvec, 4 * num_bands__, 1);

    #ifdef __GPU
    if (pu == GPU) {
        phi.component(0).pw_coeffs().allocate_on_device();
        phi.component(0).pw_coeffs().copy_to_device(0, 4 * num_bands__);
        hphi.component(0).pw_coeffs().allocate_on_device();
    }
    #endif
    
    mpi_comm_world().barrier();
    sddk::timer t1("h_loc");
    for (int i = 0; i < 4; i++) {
        hloc.apply_h(0, phi, hphi, i * num_bands__, num_bands__);
    }
    mpi_comm_world().barrier();
    t1.stop();

    #ifdef __GPU
    if (pu == GPU && !phi.component(0).pw_coeffs().is_remapped()) {
        hphi.component(0).pw_coeffs().copy_to_host(0, 4 * num_bands__);
    }
    #endif

    auto cs1 = phi.component(0).checksum_pw(0, 4 * num_bands__, CPU);
    auto cs2 = hphi.component(0).checksum_pw(0, 4 * num_bands__, CPU);

    std::cout << "checksum(phi): " << cs1 << " checksum(hphi): " << cs2 << std::endl;

    double diff{0};
    for (int i = 0; i < 4 * num_bands__; i++) {
        for (int j = 0; j < phi.component(0).pw_coeffs().num_rows_loc(); j++) {
            diff += std::pow(std::abs(2.71828 * phi.component(0).pw_coeffs().prime(j, i) - hphi.component(0).pw_coeffs().prime(j, i)), 2);
        }
    }
    if (diff != diff) {
        TERMINATE("NaN");
    }
    mpi_comm_world().allreduce(&diff, 1);
    diff = std::sqrt(diff / 4 / num_bands__ / gvec.num_gvec());
    if (mpi_comm_world().rank() == 0) {
        printf("RMS: %18.16f\n", diff);
    }
    if (diff > 1e-14) {
        TERMINATE("RMS is too large");
    }

    fft.dismiss();
}

int main(int argn, char** argv)
{
    cmd_args args;
    args.register_key("--mpi_grid_dims=", "{int int} dimensions of MPI grid");
    args.register_key("--cutoff=", "{double} wave-functions cutoff");
    args.register_key("--reduce_gvec=", "{int} 0: use full set of G-vectors, 1: use reduced set of G-vectors");
    args.register_key("--num_bands=", "{int} number of bands");
    args.register_key("--use_gpu=", "{int} 0: CPU only, 1: hybrid CPU+GPU");
    args.register_key("--gpu_ptr=", "{int} 0: start from CPU, 1: start from GPU");
    args.register_key("--repeat=", "{int} number of repetitions");

    args.parse_args(argn, argv);
    if (args.exist("help")) {
        printf("Usage: %s [options]\n", argv[0]);
        args.print_help();
        return 0;
    }
    auto mpi_grid_dims = args.value< std::vector<int> >("mpi_grid_dims", {1, 1});
    auto cutoff = args.value<double>("cutoff", 2.0);
    auto reduce_gvec = args.value<int>("reduce_gvec", 0);
    auto num_bands = args.value<int>("num_bands", 10);
    auto use_gpu = args.value<int>("use_gpu", 0);
    auto gpu_ptr = args.value<int>("gpu_ptr", 0);
    auto repeat = args.value<int>("repeat", 3);

    sirius::initialize(1);
    for (int i = 0; i < repeat; i++) {
        test_hloc(mpi_grid_dims, cutoff, num_bands, reduce_gvec, use_gpu, gpu_ptr);
    }
    mpi_comm_world().barrier();
    sddk::timer::print();
    //runtime::Timer::print_all();
    sirius::finalize();
}
