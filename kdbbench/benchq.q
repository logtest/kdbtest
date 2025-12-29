\p 12288
\c 33 333
EMA:{[x;n] ema[2%(n+1);x]};
MACD:{[x;nFast;nSlow;nSig] diff:EMA[x;nFast]-EMA[x;nSlow]; sig:EMA[diff;nSig]; :diff - sig;};

gendata:{[n;nsym]
 times:([] time: 2025.01.01 + 0D00:05:00 *til `int$n%nsym);
 symbols:nsym# distinct ([] sym:n?`3);
 d:`sym`time xasc symbols cross times;
 d: update open:100+sums -0.5+n?1f, volume:n?1000 from d;
 d: update high:open+n?1f, low:open-n?1f from d;
 d: update close:0.5*high+low, turnover:open*volume from d;
 :d;
 };

numdata:10000000; //change this value for different size test

if[0<count .z.x;numdata:: `long$"I"$.z.x 0];
 
timeloaddata:value "\\t d:gendata[",(string numdata),";500]";

timesumtil: value "\\t do[10;sum til ",(string numdata),"]";
timebysym: value "\\t select first open, min low, max high, last close, sum volume, turnover:sum volume*close by sym from d";
timebysym2nd :value "\\t select first open, min low, max high, last close, sum volume, turnover:sum volume*close by sym from d";

timebysymtime:value "\\t select first open, min low, max high, last close, sum volume, turnover:sum volume*close by sym, 0D01:00:00 xbar time from d";
timebysymtime2:value "\\t select first open, min low, max high, last close, sum volume, turnover:sum volume*close by sym, 0D01:00:00 xbar time from d";
timebysymsignal:value "\\t update ma30:EMA[close;30], MACD:MACD[close;15;30;15] by sym from d";
timebysymsignal2:value "\\t update ma30:EMA[close;30], MACD:MACD[close;15;30;15] by sym from d";
timesort: value "\\t d:`sym`time xasc d";

d1:-10000#d;

timeupsert: value "\\t do[1000;`d upsert d1]";
d:`sym`time xasc `sym`time xkey numdata#d;
timeupsertsorted: value "\\t `d upsert d1";

symlist:100#exec distinct sym from d;
timeselect: value "\\t select from d where sym in symlist";
timeselect2: value "\\t select first open, min low, max high, last close, sum volume, turnover:sum volume*close by sym from d where sym in symlist";


result:enlist `loaddata`num`sum`bysym`bysym2`bysymtime`bysymtime2`bysymsignal`bysymsignal2`timesort`timeupsert`timeupsertsorted`timeselect`timeselect2!(timeloaddata,numdata,timesumtil,timebysym,timebysym2nd,timebysymtime,timebysymtime2,timebysymsignal,timebysymsignal2,timesort,timeupsert,timeupsertsorted,timeselect,timeselect2);
show result
