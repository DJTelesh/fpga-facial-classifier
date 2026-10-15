`timescale 1ns/1ps

module tb_mac_unit;

    localparam int DW = 8;
    localparam int AW = 32;

    // 1. one signal per MAC port, same widths and signedness
    logic                 clk;
    logic                 rst_n;
    logic                 en;
    logic                 lb;
    logic signed [DW-1:0] W1_q;
    logic signed [DW-1:0] w_q;
    logic signed [AW-1:0] b1_q;
    logic signed [AW-1:0] acc1;

    // 2. the design under test, ports connected by name
    mac_unit #(.DATA_WIDTH(DW), .ACC_WIDTH(AW)) dut (
        .clk   (clk),
        .rst_n (rst_n),
        .en    (en),
        .lb    (lb),
        .W1_q  (W1_q),
        .w_q   (w_q),
        .b1_q  (b1_q),
        .acc1  (acc1)
    );

    // 3. 50 MHz clock
    initial clk = 0;
    always #10 clk = ~clk;

    // 4. the test
    initial begin
        // set every input to a known value, hold reset low
        // release reset
        // one cycle: en + lb, bias = 100
        // three cycles: en, one weight/input pair each
        // drop en, wait a cycle
        // print acc1, compare to expected, print PASS or FAIL
        $finish;
    end

endmodule