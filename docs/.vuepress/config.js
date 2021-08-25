module.exports = {
    lang: 'zh-CN',
    title: "ADMP's Manual",
    description: 'Automatic Differentiation Multipole Moment Molecular Forcefield',

    plugins: ['code-switcher',],


    markdown: {
      extendMarkdown: md => {
        md.set({ html: true });
        md.use(require('markdown-it-katex'))
        md.use(require('markdown-it-imsize'))
      },
      lineNumbers: true
    },

    head: [
      ['meta', { name: 'viewport', content: 'width=device-width,initial-scale=1,user-scalable=no' }],
      ['link', {
        rel: 'stylesheet',
        href: 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.5.1/katex.min.css'
      }],
    ],
    theme: 'reco',
    themeConfig: {
      nav: [
        { text: 'Home', link: '/' },
        { text: 'Manual', link: '/man/' },
      ],
      noFoundPageByTencent: false,
      repo: 'https://github.com/Roy-Kid/ADMP',
      sidebar:[
        '/man/',
        ['/man/man1', '开发指南'],
      ]
    }
  
  }

