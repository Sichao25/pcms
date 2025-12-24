#include <catch2/catch_test_macros.hpp>
#include "pcms/assert.h"
#include "pcms/print.h"
#include <iostream>

int raise_error(int code)
{
  if (code) {
    throw pcms::exception("Test exception", code, "Raising error for testing");
    return 1;
  }
  return 0;
}

TEST_CASE("pcms error handling test")
{
  REQUIRE_THROWS_AS(raise_error(1), pcms::exception);
}