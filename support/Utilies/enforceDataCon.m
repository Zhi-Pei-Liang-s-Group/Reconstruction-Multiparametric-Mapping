function sxt_dataCon = enforceDataCon(sxt_ref,sxt_measure,kmask)

skt_ref = F3_x2k(sxt_ref);
skt_measure = F3_x2k(sxt_measure);
skt_dataCon = bsxfun(@times,skt_ref,~kmask) + bsxfun(@times,skt_measure,kmask);
sxt_dataCon = F3_k2x(skt_dataCon);