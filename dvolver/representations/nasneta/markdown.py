def generate_markdown_table(individuals):
    """
    Generates markdown table with individuals architecture, accuracy and speed
    """
    def gen_header_block(i, type):
        return 'b%s%d.h_left|b%s%d.op_left|b%s%d.h_right|b%s%d.op_right|'%(type, i, type, i, type, i, type, i)

    def gen_header(nb_blocks):
        md = '|id|accuracy|speed|'

        for i in range(nb_blocks//2):
            md += gen_header_block(i, 'n') # header for normal cell

        md += 'normal_concat|'

        for i in range(nb_blocks//2):
            md += gen_header_block(i, 'r') # header for reduction cell

        md += 'reduction_concat|'

        md += '\n'

        md += '|:---:|:---:|:---:|'

        for i in range(nb_blocks//2):
            md += ':---:|:---:|:---:|:---:|'

        md += ':---:|'

        for i in range(nb_blocks//2):
            md += ':---:|:---:|:---:|:---:|'

        md += ':---:|'

        md += '\n'

        return md

    def gen_md(individual):

        md = '|%s|'%(str(individual.archIndex))

        md += '%f|%f|'%(individual.fitness.values[1], individual.fitness.values[0])

        nb_blocks = (len(individual) - 2) // 4

        for i in range(nb_blocks//2):
            md += '%d|%s|%d|%s|'%(individual[4*i], individual[4*i+1], individual[4*i+2], individual[4*i+3])

        md += '%s|'%(str(individual[4*nb_blocks//2]))

        for i in range(nb_blocks//2):
            md += '%d|%s|%d|%s|'%(individual[4*nb_blocks//2 + 1 + 4*i], individual[4*nb_blocks//2 + 1 + 4*i+1], individual[4*nb_blocks//2 + 1 + 4*i+2], individual[4*nb_blocks//2 + 1 + 4*i+3])

        md += '%s|'%(str(individual[8*nb_blocks//2 + 1]))

        md += '\n'

        return md


    if len(individuals) == 0:
        return ''

    nb_blocks = (len(individuals[0]) - 2) // 4

    assert nb_blocks % 2 == 0

    return ''.join([gen_header(nb_blocks)] + [gen_md(individual) for individual in individuals])
