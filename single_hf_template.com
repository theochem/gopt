%chk=${title}.chk
#p b3lyp/6-31G ${freq} SCF(XQC)

${title}

${charge} ${multi}
${atoms}



