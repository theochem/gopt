%chk=${title}.chk
#p hf/6-31+G ${freq} SCF(XQC) nosymmetry

${title}

${charge} ${multi}
${atoms}



