#include <catch2/catch_test_macros.hpp>
#include "pcms/assert.h"
#include "pcms/print.h"
#include <iostream>

int raise_error(int code)
{
  if (code) {
    pcms::handle_error(pcms::exception("Mesh validation failed", 1,
                                       pcms::error_level::recoverable));

    return 1;
  }
  return 0;
}

TEST_CASE("pcms error handling test")
{
  REQUIRE_THROWS_AS(raise_error(1), pcms::exception);
}