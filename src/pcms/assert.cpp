#include "pcms/assert.h"
#include "pcms/print.h"
#include <cstdio>
#include <cstdlib>
namespace pcms
{

namespace
{
error_mode current_mode = error_mode::throw_exception;
}

void set_error_mode(error_mode mode)
{
  current_mode = mode;
}

void handle_error(const exception& e)
{
  switch (current_mode) {
    case error_mode::throw_exception: throw e;

    case error_mode::abort: std::abort();

    case error_mode::return_code:
    default: break;
  }
}

void handle_mpi_error(const pcms::exception& e)
{
  int local_fatal = (e.level() == pcms::error_level::fatal);
  int global_fatal = 0;

  MPI_Allreduce(&local_fatal, &global_fatal, 1, MPI_INT, MPI_MAX,
                MPI_COMM_WORLD);

  switch (current_mode) {
    case pcms::error_mode::throw_exception: throw e;

    case pcms::error_mode::abort:
      if (global_fatal)
        MPI_Abort(MPI_COMM_WORLD, e.code());
      break;

    case pcms::error_mode::return_code:
    default: break;
  }
}

void Pcms_Assert_Fail(const char* msg)
{
  printError("%s", msg);
  abort();
}
} // namespace pcms
