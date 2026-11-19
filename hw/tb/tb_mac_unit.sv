// Run from the REPO ROOT (the $readmemh paths are relative to it):
//   iverilog -g2012 -o tb_mac_unit.vvp hw/rtl/mac_unit.sv hw/tb/tb_mac_unit.sv
//   vvp -n tb_mac_unit.vvp
// int16 configuration (vectors from gen_vectors.py --bits 16):
//   iverilog -g2012 -Ptb_mac_unit.DW=16 -Ptb_mac_unit.AW=48 -o tb_mac_unit.vvp ...
`timescale 1ns/1ps
`default_nettype none

module tb_mac_unit;

    parameter int DW = 8;     // DATA_WIDTH
    parameter int AW = 32;    // ACC_WIDTH, from arithmetic_contract.md section 4
    localparam int N_IN     = 2304;
    localparam int N_HIDDEN = 64;
    localparam int N_IMAGES = 4;    // must match N_IMAGES in gen_vectors.py

    // 1. one signal per MAC port, same widths and signedness
    logic                 clk;
    logic                 rst_n;
    logic                 en;
    logic                 lb;
    logic signed [DW-1:0] w;
    logic signed [DW-1:0] x;
    logic signed [AW-1:0] bias;
    logic signed [AW-1:0] acc;

    //for python golden numbers
    logic signed [DW-1:0] w_mem   [0:N_HIDDEN*N_IN-1];
    logic signed [DW-1:0] x_mem   [0:N_IN-1];
    logic signed [AW-1:0] b_mem   [0:N_HIDDEN-1];
    logic signed [AW-1:0] exp_mem [0:N_HIDDEN-1];

    int fails = 0;          // directed-test failures
    int golden_fails = 0;   // golden neuron mismatches
    string vec_dir;         // int8 vectors live at the top, other widths in int<DW>/

    // 2. the design under test, ports connected by name
    mac_unit #(.DATA_WIDTH(DW), .ACC_WIDTH(AW)) dut (
        .clk   (clk),
        .rst_n (rst_n),
        .en    (en),
        .lb    (lb),
        .w     (w),
        .x     (x),
        .bias  (bias),
        .acc   (acc)
    );

    // Every task drives inputs on the negedge, so the MAC captures them on the
    // NEXT posedge. acc only reflects a mac() call after one more clock edge,
    // which is why every check comes after idle().
    task automatic mac(input logic signed [DW-1:0] w_in,
                       input logic signed [DW-1:0] x_in);
        @(negedge clk);
        en = 1;
        lb = 0;
        w  = w_in;
        x  = x_in;
    endtask

    task automatic idle();
        @(negedge clk);
        en = 0;
    endtask

    task automatic do_reset();
        // en/lb low first, or the MAC adds whatever is left on w/x the
        // cycle after reset releases
        en = 0;
        lb = 0;
        repeat (3) @(negedge clk) begin
            rst_n = 0;
        end
        @(negedge clk) begin
            rst_n = 1;
        end
    endtask

    task automatic load_bias(input logic signed [AW-1:0] b);
        @(negedge clk) begin
            en   = 1;
            lb   = 1;
            bias = b;
        end
    endtask

    task automatic check(input string name, input logic signed [AW-1:0] expected);
        if (acc === expected)
            $display("PASS %s: acc = %0d", name, acc);
        else begin
            $display("FAIL %s: acc = %0d, expected %0d", name, acc, expected);
            fails++;
        end
    endtask

    // 3. 50 MHz clock
    initial clk = 0;
    always #10 clk = ~clk;

    initial begin
        rst_n = 0;
        en    = 0;
        lb    = 0;
        w     = 0;
        x     = 0;
        bias  = 0;

        // test 1: accumulate from reset, including the -128 input corner
        // 5*3 + -4*7 + -128*-127 = 15 - 28 + 16256
        do_reset();
        mac(5, 3);
        mac(-4, 7);
        mac(-128, -127);
        idle();
        check("test 1 (accumulate)", 16243);

        // test 2: en=0 beats lb=1 -- the register just holds, it is not a latch
        do_reset();
        @(negedge clk) begin
            en   = 0;
            lb   = 1;
            bias = 3;
            w    = 5;
            x    = 4;
        end
        idle();
        check("test 2 (en beats lb)", 0);

        // test 3: bias then accumulate, 3 - 28 + 16256
        do_reset();
        load_bias(3);
        mac(-4, 7);
        mac(-128, -127);
        idle();
        check("test 3 (bias + accumulate)", 16231);

        // test 4: negative sum, -28 - 12
        do_reset();
        mac(-4, 7);
        mac(-2, 6);
        idle();
        check("test 4 (negative)", -40);

        // test 5: bias load alone
        do_reset();
        load_bias(1920);
        idle();
        check("test 5 (bias only)", 1920);

        // test 6, GOLDEN: every hidden neuron of every dumped image vs infer_int()
        vec_dir = (DW == 8) ? "test_vectors" : $sformatf("test_vectors/int%0d", DW);
        $display("DATA_WIDTH %0d, ACC_WIDTH %0d, vectors from %s", DW, AW, vec_dir);
        $readmemh({vec_dir, "/w1.hex"}, w_mem);
        $readmemh({vec_dir, "/b1.hex"}, b_mem);
        if ($isunknown(w_mem[0]) || $isunknown(b_mem[0]))
            $fatal(1, "test vectors not loaded -- run vsim from the repo root");

        for (int k = 0; k < N_IMAGES; k++) begin
            $readmemh($sformatf("%s/img%0d_x.hex", vec_dir, k),    x_mem);
            $readmemh($sformatf("%s/img%0d_acc1.hex", vec_dir, k), exp_mem);
            for (int j = 0; j < N_HIDDEN; j++) begin
                // no reset between neurons: load_bias restarts the sum
                load_bias(b_mem[j]);
                for (int i = 0; i < N_IN; i++) begin
                    mac(w_mem[j*N_IN+i], x_mem[i]);
                end
                idle();
                if (acc !== exp_mem[j]) begin
                    golden_fails++;
                    if (golden_fails <= 10)   // don't bury the log
                        $display("FAIL GOLDEN img %0d neuron %0d: acc = %0d, expected %0d",
                                 k, j, acc, exp_mem[j]);
                end
            end
        end
        $display("GOLDEN: %0d/%0d neurons match",
                 N_IMAGES*N_HIDDEN - golden_fails, N_IMAGES*N_HIDDEN);

        if (fails == 0 && golden_fails == 0)
            $display("ALL TESTS PASSED");
        else
            $display("*** %0d directed FAIL(s), %0d golden mismatch(es) ***",
                     fails, golden_fails);
        $finish;
    end

endmodule

`default_nettype wire
