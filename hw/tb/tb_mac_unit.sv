`timescale 1ns/1ps
`default_nettype none

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
    task automatic mac(input logic signed [DW-1:0] w,
                   input logic signed [DW-1:0] x);
    @(negedge clk);
    en   = 1;
    lb   = 0;
    W1_q = w;
    w_q  = x;
    endtask

    task automatic idle();
    @(negedge clk);
    en   = 0;
    endtask

    task automatic do_reset();
    repeat (3) @(negedge clk) begin
        rst_n = 0;
    end
    @(negedge clk) begin
        rst_n = 1;
    end
    endtask

    task automatic load_bias(input logic signed [AW-1:0] b);
    @(negedge clk) begin
        en = 1;
        lb = 1;
        b1_q = b;
    end
    endtask

    // 3. 50 MHz clock
    initial clk = 0;
    always #10 clk = ~clk;

    // One test
    initial begin

        rst_n = 0;
        en    = 0;
        lb    = 0;
        W1_q  = 0;
        w_q   = 0;
        b1_q  = 0;

        do_reset();
        mac(5, 3);
        mac(-4, 7);
        mac(-128, -127);
        if (acc1 === -13)
            $display("PASS test 1: acc1 = %0d", acc1);
        else
            $display("FAIL test 1: acc1 = %0d, expected -13", acc1);        

    // Two test
        do_reset();
        @(negedge clk) begin
            en   = 0;
            lb   = 1;
            b1_q = 3;
            W1_q = 5;
            w_q  = 4;
        end //MAC should be nothing, is this a latch?? ah oh
        load_bias(3);
        mac(-4, 7);
        mac(-128, -127);
        if (acc1 === -25)
            $display("PASS test 2: acc1 = %0d", acc1);
        else
            $display("FAIL test 2: acc1 = %0d, expected -25", acc1);        

    // Three test
        do_reset();
        @(negedge clk) begin
            en   = 0;
            lb   = 1;
            b1_q = 3;
            W1_q = 5;
            w_q  = 4;
        end //MAC should be nothing, is this a latch?? ah oh

        load_bias(3);
        mac(-4, 7);
        mac(-128, -127);
        idle();
        if (acc1 === 16231)
            $display("PASS test 3: acc1 = %0d", acc1);
        else
            $display("FAIL test 3: acc1 = %0d, expected 16231", acc1);        

        //test 4
        do_reset();
        mac(-4, 7);
        mac(-2, 6);
        idle();
        if (acc1 === -40)
            $display("PASS test 4: acc1 = %0d", acc1);
        else
            $display("FAIL test 4: acc1 = %0d, expected -40", acc1);   

        //test 5
        do_reset();
        load_bias(1920);
        idle();
        if (acc1 === 1920)
            $display("PASS test 5: acc1 = %0d", acc1);
        else
            $display("FAIL test 5: acc1 = %0d, expected 1920", acc1);   
    $finish;
    end

endmodule